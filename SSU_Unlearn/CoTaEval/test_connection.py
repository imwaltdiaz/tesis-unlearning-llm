import argparse
import os
import sys
import redis
import dataportraits
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--start-from-dir", type=str, help="Start redis, load all bloom filters from a directory", default=None)
    command_group.add_argument("--just-start", action='store_true', help="Start redis and don't do anything else")
    command_group.add_argument("--dump-to-dir", type=str, help="Dump all filters from a redis instance to a directory", default=None)
    command_group.add_argument("--shutdown-redis", action='store_true', help="Shutdown redis instance")

    parser.add_argument("--host", type=str, help="redis host", default='localhost')
    parser.add_argument("--port", type=int, help="redis port", default=6379)
    parser.add_argument("--allow-key-overwrites", action='store_true', help="Allow overwriting the redis keys")
    parser.add_argument("--no-create-dirs", action='store_true', help="Don't create the directory to write to")

    args = parser.parse_args()

    n = 6
    bloom_filter = "gutenberg_books_time_step_1_tokenized.36-36.bf"
    print("Test Bloom Filter name is:", bloom_filter)
    portrait = dataportraits.RedisBFSketch('localhost', 6379, bloom_filter, int(n))
    print("Successfully loaded testing chunks from", bloom_filter)
