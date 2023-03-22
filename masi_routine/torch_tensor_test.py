import torch
import sys


def main():
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([10, 20, 30, 40, 50])
    mask = torch.tensor([0, 1, 0, 1, 0], dtype=torch.uint8)
    a[mask == 1] = b[mask == 1][:]
    print(a)


if __name__ == "__main__":
    sys.exit(main())