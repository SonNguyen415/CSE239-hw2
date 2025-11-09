import rpyc 
import string 
import collections 
import itertools 
import time 
import operator 
import glob 
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed 
import socket
import threading

# ANSI color codes for terminal output
RESET = '\033[0m'
RED = '\033[91m'

PORT = 18861
SERVICE_NAME = 'worker'
TIMEOUT = 20
MAX_RETRIES = 4 # Max retries for worker discovery
INPUT_DIR = 'txt'
MAX_CHUNK_SZ = 32 * 1024 * 1024 # 32 MB max chunk size 

class WorkerPool:
    """Thread-safe worker pool with connection management"""
    def __init__(self, worker_ips: list):
        # Lock and condition variable for thread safety
        self.lock = threading.Lock()
        self.worker_available = threading.Condition(self.lock)

        # A set of available and failed workers, each as a worker_ip
        # Probably better to use a queue for available workers, but set with lock is simpler for now.
        self.available_workers = set(worker_ips)
        self.current_workers = set()

    def connect_to_worker(self, worker_ip: str, chunk_idx: int):
        """
        Connect to a single worker. On failure, simply return None.
        """
        try:
            with self.lock:
                conn = rpyc.connect(worker_ip, PORT, config={"sync_request_timeout": TIMEOUT, "allow_pickle": True})
                self.current_workers.add(worker_ip)
            return conn
        except Exception as e:
            print(f"{RED}Failed to connect to worker at {worker_ip}:{PORT} - {e}{RESET}", flush=True)
            return None

    def get_worker(self):
        """
        Return a free worker ip, raises RuntimeError if none available.
        Block until a worker is free and available.
        """
        with self.lock:
            while not self.available_workers:
                if not self.current_workers and not self.available_workers:
                    return None
                self.worker_available.wait()
            return self.available_workers.pop()

    def mark_failed(self, worker_ip: str, conn: rpyc.Connection):
        """Mark a worker as failed and try to close connection"""
        with self.lock:
            try:
                self.current_workers.discard(worker_ip)
                conn.close()
            except:
                pass
            print(f"{RED}Worker {worker_ip} marked as failed{RESET}", flush=True)

            # All workers have failed, raise error and notify the other threads
            if not self.current_workers and not self.available_workers:
                self.worker_available.notify_all()
                raise RuntimeError("All workers have failed")

    def return_worker(self, worker_ip: str, conn: rpyc.Connection):
        """Return a worker to the available pool"""
        with self.lock:
            conn.close()
            self.current_workers.discard(worker_ip)
            self.available_workers.add(worker_ip)
            self.worker_available.notify()

def discover_workers():
    """
    Discover all worker containers via DNS with retries.
    Returns:
        list: List of worker IP addresses
    """
    for attempt in range(MAX_RETRIES):
        try:
            addr_info = socket.getaddrinfo(SERVICE_NAME, PORT, 
                                           socket.AF_INET, socket.SOCK_STREAM)
            workers = list(set([info[4][0] for info in addr_info]))
            if workers:
                return workers
        except socket.gaierror:
            pass
        print(f"Waiting for workers... (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
        time.sleep(2)
    print(f"{RED}No workers discovered after multiple retries{RESET}")
    sys.exit(1)


def process_chunk(chunk: str, chunk_idx: int, worker_pool: WorkerPool, fn="map"):
    """Process a chunk. Fn is either 'map' or 'reduce'."""
    while ip := worker_pool.get_worker():
        # Loop until success or no workers available
        try:
            conn = worker_pool.connect_to_worker(ip, chunk_idx)
            if not conn: # Failed to connect, try again with a different worker
                continue 
            result = conn.root.map(chunk) if fn == "map" else conn.root.reduce(chunk)
            result = rpyc.utils.classic.obtain(result)
            worker_pool.return_worker(ip, conn)
            return result
        except Exception as e:
            print(f"{RED}Error processing chunk {chunk_idx} on worker {ip}: {e}{RESET}", flush=True)
            worker_pool.mark_failed(ip, conn)

    print(f"{RED}No workers available for chunk {chunk_idx}. Exiting.{RESET}", flush=True)
    sys.exit(1)


def read_and_process_chunk(file: str, chunk_idx: int, worker_pool: WorkerPool):
    """Read a chunk from a single file. Max size is MAX_CHUNK_SZ."""
    file_size = os.path.getsize(file)
    start = chunk_idx * MAX_CHUNK_SZ
    if start >= file_size:
        return None
    end = min(start + MAX_CHUNK_SZ, file_size)
    
    with open(file, 'rb') as f:
        f.seek(start)
        if chunk_idx > 0:
            f.readline()
            start = f.tell()
            if start >= file_size:
                return None
        chunk_bytes = f.read(end - start)
        if end < file_size:
            chunk_bytes += f.readline()
        chunk = chunk_bytes.decode('utf-8', errors='ignore')
    return process_chunk(chunk, chunk_idx, worker_pool, fn="map")
    
def mapreduce_wordcount(texts: list): 
    """
    Performs a distributed MapReduce word count over the input texts.
    """
    # 1. Discover workers and create worker pool
    worker_ips = discover_workers()
    worker_pool = WorkerPool(worker_ips)
    nworkers = len(worker_ips)

    # 2. Split files into chunks
    tasks = []
    for file in texts:
        file_size = os.path.getsize(file)
        num_chunks = (file_size + MAX_CHUNK_SZ - 1) // MAX_CHUNK_SZ
        for chunk_idx in range(num_chunks):
            tasks.append((file, chunk_idx))

    # 3. MAP PHASE: Send chunks and get intermediate pairs
    map_results = []
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        # I don't think thread pool is parallel but workers are so it should be fine
        futures = [executor.submit(read_and_process_chunk, file, i, worker_pool) 
                for file, i in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
                map_results.append(result)
            except Exception as e:
                print(f"{RED}Error processing map task: {e}{RESET}", flush=True)

    # 4. SHUFFLE PHASE: Partition and group by key
    reduce_inputs = partition_dict(map_results, nworkers)
    
    # 5. REDUCE PHASE: Send grouped data to reducers
    total_counts = collections.Counter()
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        futures = [executor.submit(process_chunk, reduce_inputs[i], i, worker_pool, fn="reduce") 
                for i in range(len(reduce_inputs))]
        for future in as_completed(futures):
            try:
                result = future.result()
                total_counts.update(result)
            except Exception as e:
                print(f"{RED}Error processing reduce task: {e}{RESET}", flush=True)

    # 6. FINAL AGGREGATION
    return sorted(total_counts.items(), key=operator.itemgetter(1), reverse=True)


def partition_dict(map_results: list, n: int):
    """
    Partition and group map results across n reducers.
    
    Args:
        map_results: list of dicts from mappers [{word: count}, {word: count}, ...]
        n: number of partitions (reducers)
    
    Returns:
        List of n dicts, each containing {word: [count1, count2, ...]}
    """
    partitions = [collections.defaultdict(list) for _ in range(n)]
    for map_output in map_results:
        for word, count in map_output.items():
            partition_idx = hash(word) % n
            partitions[partition_idx][word].append(count)
    
    # Convert defaultdict to regular dict
    return [dict(partition) for partition in partitions]
    

def download(urls=['http://mattmahoney.net/dc/enwik9.zip']):
    os.makedirs(INPUT_DIR, exist_ok=True)
    zip_path = os.path.join(INPUT_DIR, 'tmp.zip')
    for url in urls:    
        print(f"Downloading {url}...", flush=True)
        os.system(f"curl -o {zip_path} {url}")
        os.system(f"unzip -o {zip_path} -d {INPUT_DIR}")
        os.system(f"rm {zip_path}")
        os.system(f"cd {INPUT_DIR}; ls")


if __name__ == "__main__": 
    # DOWNLOAD AND UNZIP DATASET 
    if len(sys.argv) > 1:
        text = download(sys.argv[1:])
    else:
        text = download()

    start_time = time.time() 
    input_files = glob.glob(f'{INPUT_DIR}/*') 
    word_counts = mapreduce_wordcount(input_files) 
    
    print('\nTOP 20 WORDS BY FREQUENCY\n', flush=True) 
    top20 = word_counts[0:20] 
    longest = max(len(word) for word, count in top20) 
    i = 1
    for word, count in top20:
        print('%s.\t%-*s: %5s' % (i, longest+1, word, count), flush=True)
        i = i + 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time: {} seconds".format(elapsed_time), flush=True)
