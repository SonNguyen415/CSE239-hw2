# Advanced Cloud Computing: Homework 2

### Instructions

Download the git repo:
```sh
git clone https://github.com/SonNguyen415/CSE239-hw2.git
```

To build with docker:
```sh
docker compose build
```

Run with variable number of workers:
```sh
docker compose up --scale worker=<number of workers>
```

To change the dataset used, update the command-line argument:
```sh
command: ["python3", "coordinator.py", <your_desired_url>]
```
