#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <typeinfo>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// vector<int> arr = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}; // どっちでも同じ結果
vector<int> arr = {{1, -1, -1}}; // どっちでも同じ結果

// int findIndex2(const vector<int> &arr, int item) {
//     auto ret = std::find(arr.begin(), arr.end(), item);

//     if (ret != arr.end()) return ret - arr.begin();
//     return -1;
// }
int findIndex2() {
    int item = -1; // 10;
    auto ret = std::find(arr.begin(), arr.end(), item);

    if (ret != arr.end()) return ret - arr.begin();
    // return -1;
    return -100;
}

// int main(int argc, char *argv[]) {
int main() {
    // vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // vector<int> arr = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}; // どっちでも同じ結果

    // auto pos2 = findIndex2(arr, 10);
    auto pos2 = findIndex2();

    // どちらも同じ
    // pos2 != -1
    pos2 != -100
        ? cout << "Found the element " << 10 << " at position " << pos2 << endl
        : cout << "Could not found the element " << 10 << " in vector" << endl;
    // if(pos2 != -1){
    if(pos2 != -100){
        cout << "Found the element " << 10 << " at position " << pos2 << endl;
    }else{
        cout << "Could not found the element " << 10 << " in vector" << endl;
    }
    cout << typeid(pos2).name() << endl; // pos2 = int

    exit(EXIT_SUCCESS);
}