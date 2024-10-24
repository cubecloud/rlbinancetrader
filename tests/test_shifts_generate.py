from typing import Union, List, Tuple


def generate_shifts(num_shifts: int) -> List[int]:
    def half_list(half: List[int]) -> List[int]:
        result: list = []
        if len(half) <= 2:
            half.reverse()
            result.extend(half)
        else:
            mid = len(half) // 2
            if mid > 0:
                result.append(half[mid])
                result.extend(half_list(list(range(half[0], half[mid]))))
                result.extend(half_list(list(range(half[mid + 1], half[-1] + 1))))
        return result

    shifts_lst = [0]
    mid = num_shifts // 2
    shifts_lst.append(mid)
    first_half = half_list(list(range(1, mid)))
    second_half = half_list(list(range(mid + 1, num_shifts)))
    max_len = min(len(first_half), len(second_half))
    for a, b in zip(first_half, second_half):
        shifts_lst.extend([a, b])
    shifts_lst.extend(first_half[max_len:])
    shifts_lst.extend(second_half[max_len:])
    return shifts_lst


if __name__ == '__main__':
    s = generate_shifts(60)
    print(s)
    print(len(s))
