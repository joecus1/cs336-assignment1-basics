import os
from typing import BinaryIO
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_file_pre_tokens(filepath: str) -> dict[tuple[bytes], int]:
    ## Usage
    with open(filepath, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        token_counts = {}

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            
            # for match in re.finditer(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", chunk):
            for token in chunk.split():
                # token = match.group()
                token_bytes = tuple(x.encode('utf-8') for x in token)
                token_counts[token_bytes] = token_counts.get(token_bytes, 0) + 1
            
        return token_counts


def byte_tuple_to_str(token_tuple: tuple[bytes, ...]) -> str:
    return b''.join(token_tuple).decode('utf-8')


def token_dict_to_human_readable(token_dict: dict[tuple[bytes, ...], int]) -> dict[str, int]:
    return {byte_tuple_to_str(k): v for k, v in token_dict.items()}


def count_byte_pairs(token_dict: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for token_tuple, count in token_dict.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def join_pairs_in_token_tuple(token_tuple: tuple[bytes, ...], pair_to_join: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    token_list = []
    i = 0
    while i < len(token_tuple):
        if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == pair_to_join:
            token_list.append(token_tuple[i] + token_tuple[i + 1])
            i += 2
        else:
            token_list.append(token_tuple[i])
            i += 1
    return tuple(token_list)


def join_pairs_in_token_dict(
    token_dict: dict[tuple[bytes, ...], int],
    most_common_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    return {join_pairs_in_token_tuple(k, most_common_pair): v
            for k, v in token_dict.items()}
    


if __name__ == "__main__":
    token_dict = count_file_pre_tokens("data/bpe_training_example.txt")
    print("Pre-token counts:")
    print(token_dict)
    
    byte_pairs = count_byte_pairs(token_dict)
    print("Byte pair counts:")
    print(byte_pairs)
    
    most_common_pair = max(byte_pairs.items(), key=lambda x: (x[1], x[0]))[0]
    print("Most common byte pair:", byte_tuple_to_str(most_common_pair))
    
    joined_dict = join_pairs_in_token_dict(token_dict, most_common_pair)
    print("Dict with pairs joined:", joined_dict)
    