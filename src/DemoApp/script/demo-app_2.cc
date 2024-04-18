#include <bits/stdc++.h>
using namespace std;
// using std::fill;

#define SIZE_OF_ARRAY(results_data)    (sizeof(results_data)/sizeof(results_data[0]))

int main() {

    int results_data[] = {-1, -1, -1}; // 今回は過去3つの結果を保存
    
    cout << results_data << endl; // 未定義の動作

    // cout << sizeof(results_data) << endl; // アドレス=8byte　指す先はint=4byte

    for(int i=0; i<3; i++){
        results_data[i] = i;
        // cout << *results_data << endl;
        cout << results_data[i] << endl;
    }

    cout << "*****" << endl;

    bool is_hsj = true;
    
    for(int i=0; i<3; i++){
        if(results_data[i] == i){
            cout << "True" << endl;
        }else{
            is_hsj = false;
        }
    }

    if(is_hsj){
        cout << "hop step jump !" << endl;
        // results_data = {-1, -1, -1}; // これだと別の配列を作っているだけ
        // int results_data[] = {-1};
        // results_data.fill(0);


        // for(int i=0; i<3; i++){
        //     results_data[i] = -1;
        //     cout << results_data[i] << endl;
        // }

        // " どっちでもOK "
        for (size_t i = 0; i < SIZE_OF_ARRAY(results_data); ++i) {
            results_data[i] = -1;
            cout << results_data[i] << endl;
        }

        // ラムダ式を使用して配列の各要素に操作を適用
        for_each(begin(results_data), end(results_data), [](int &n){ n *= 0; });
        // " どっちでもOK "

    }

    for(int i=0; i<3; i++){
        if(results_data[i] == i){
            cout << "True" << endl;
            cout << results_data[i] << endl;
        }else{
            is_hsj = false;
            cout << results_data[i] << endl;
        }
    }

}
