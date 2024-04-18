#include <bits/stdc++.h>
using namespace std;
// using std::fill;
#include <array>
#include <vector>

#define SIZE_OF_ARRAY(results_data)    (sizeof(results_data)/sizeof(results_data[0]))

// int results_data[] = {-1, -1, -1}; // 今回は過去3つの結果を保存

bool demo_func(std::array<int, 3>& results_data) { // intポインタで配列の先頭要素を受け取る

    // int results_data[] = {-1, -1, -1}; // 今回は過去3つの結果を保存

    // このままだと配列のサイズ情報が受け取れていない
    cout << "size : " << sizeof(results_data) << endl; // 8byte
    cout << "size : " << SIZE_OF_ARRAY(results_data) << endl; // 2byte (8/4=2)...(先頭アドレスのサイズ/指す先の中身のサイズ)

    for(int i=0; i<SIZE_OF_ARRAY(results_data); i++){
        results_data[i] = i;
        cout << results_data[i] << endl;
    }

    cout << "*****" << endl;

    bool is_hsj = true;
    
    for(int i=0; i<SIZE_OF_ARRAY(results_data); i++){
        if(results_data[i] == i){
            cout << "True" << endl;
        }else{
            is_hsj = false;
        }
    }

    if(is_hsj){
        cout << "hop step jump !" << endl;
        cout << "*****" << endl;

        for (size_t i = 0; i < SIZE_OF_ARRAY(results_data); ++i) {
            results_data[i] = -1;
            cout << results_data[i] << endl;
        }

        // ラムダ式を使用して配列の各要素に操作を適用
        // for_each(begin(results_data), end(results_data), [](int &n){ n *= 0; });

    }

    return is_hsj;

}

int main(void){
    
    // constexpr std::size_t num = 3;
    // int results_data[num] = {-1, -1, -1}; // 今回は過去3つの結果を保存

    std::array<int, 3> results_data = {{-1, -1, -1}};
    //std::array<int, 3> x = {-1, -1. -1}; // C++14～
    auto size = results_data.size();  // 3
    cout << "x size : " << size << endl;
    //auto size = std::size(x);  // C++17～

    cout << "size : " << sizeof(results_data) << endl; // 12byte
    cout << "size : " << SIZE_OF_ARRAY(results_data) << endl; // 3byte

    bool judge;
    judge = demo_func(results_data); // &resulus_data[0]
    if(judge) cout << "Guidance >> Judge == hop step jump !" << endl;

}
