from typing import List

def partition(arr: List[int], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[high], arr[i + 1] = arr[i + 1], arr[high]
    return i + 1

def quicksort(arr: List, low: int, high: int):
    if low < high:
        pivot = partition(arr, low, high)
        quicksort(arr, low, pivot - 1)
        quicksort(arr, pivot + 1, high)

arr = [6, 2, 7, 1, 3]

print("Original array: ", arr)
quicksort(arr, 0, len(arr) - 1)
print("Soretd array: ", arr)

