import rpyc 
import socket
import string
import collections
import time
class MapReduceService(rpyc.Service): 
    def exposed_map(self, chunk: str): 
        """
        Map step: tokenize and count words in text chunk.
        Args:
            chunk: text string
        Returns:
            collections.Counter: word count dictionary
        """ 
        STOP_WORDS = set([ 
        'a', 'an', 'and', 'are', 'as', 'be', 'by', 'for', 'if', 'in',  
        'is', 'it', 'of', 'or', 'py', 'rst', 'that', 'the', 'to', 'with',  
        ]) 
        TR = "".maketrans(string.punctuation, ' ' * len(string.punctuation)) 
        output = collections.Counter()
        for line in chunk.splitlines():
            if line.lstrip().startswith('..'): # Skip rst comment lines 
                continue
            line = line.translate(TR) # Strip punctuation 
            for word in line.split(): 
                word = word.lower() 
                if word.isalpha() and word not in STOP_WORDS: 
                    output[word] += 1 
        return output
    
    def exposed_reduce(self, grouped_items): 
        """
        Reduce step: sum counts for each word.
        Args:
            grouped_items: dict of {word: [count1, count2, ...]}
        Returns:
            dict: {word: total_count}
        """ 
        items = rpyc.utils.classic.obtain(grouped_items)
        result = {}
        for word, counts in items.items():
            result[word] = sum(counts)  
        
        return result


if __name__ == "__main__": 
    from rpyc.utils.server import ThreadedServer 
    hostname = socket.gethostname()
    print(f"Worker running with hostname: {hostname}")
    t = ThreadedServer(MapReduceService, port=18861, protocol_config={"allow_pickle": True}) 
    t.start() 

