#include <bits/stdc++.h>
using namespace std;
// using std::fill;

#define SIZE_OF_ARRAY(results_data)    (sizeof(results_data)/sizeof(results_data[0]))

// int results_data[] = {-1, -1, -1}; // 今回は過去3つの結果を保存

bool demo_func(int *results_data,     std::size_t num) { // intポインタで配列の先頭要素を受け取る

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

    // for(int i=0; i<SIZE_OF_ARRAY(results_data); i++){
    //     if(results_data[i] == i){
    //         cout << "True" << endl;
    //         cout << results_data[i] << endl;
    //     }else{
    //         is_hsj = false;
    //         cout << results_data[i] << endl;
    //     }
    // }

    return is_hsj;

}

int main(void){
    
    constexpr std::size_t num = 3;
    int results_data[num] = {-1, -1, -1}; // 今回は過去3つの結果を保存

    cout << "size : " << sizeof(results_data) << endl; // 12byte
    cout << "size : " << SIZE_OF_ARRAY(results_data) << endl; // 3byte

    bool judge;
    judge = demo_func(results_data, num); // &resulus_data[0]
    if(judge) cout << "Guidance >> Judge == hop step jump !" << endl;

}
