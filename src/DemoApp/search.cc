#include <iostream>
#include <algorithm>
#include <vector>
 
int main()
{
    // int arr[] = {1, 3, 5, 7, 9};
    std::vector<int> arr = {{1, 3, 5, 7, 9}}; // テスト
    int target = 7;
 
    auto it = std::find(std::begin(arr), std::end(arr), target);
 
    if (it != std::end(arr)) {
        int index = std::distance(arr, it);
        std::cout << "Element found at index " << index << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }

    int test_index = std::distance(arr, it);
    std::cout << test_index << std::endl;
 
    return 0;
}