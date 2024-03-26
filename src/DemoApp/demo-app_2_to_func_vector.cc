#include <bits/stdc++.h>
using namespace std;
#include <array>
#include <vector>

#define SIZE_OF_ARRAY(results_data)    results_data.size()  // 3
// #define MATCH "True"
#define Guidance "Hop Step Jump !"
#define LINE "********************"

#include "test_demo.h"


namespace apps {
  namespace sensor {
    bool demo_func(std::vector<int>& results_data) { // intポインタで配列の先頭要素を受け取る

        size_t size = results_data.size();  // 3
        cout << "SIZE_T : " << size << endl;

        // // このままだと配列のサイズ情報が受け取れていない
        // cout << "size : " << sizeof(results_data) << endl; // 24byte
        // cout << "size : " << SIZE_OF_ARRAY(results_data) << endl; // 6byte (8/4=2)...(先頭アドレスのサイズ/指す先の中身のサイズ)

        for(int i=0; i<SIZE_OF_ARRAY(results_data); i++){
            results_data[i] = i;
            cout << results_data[i] << endl;
        }

        cout << LINE << endl;

        bool is_hsj = true;
        
        for(int i=0; i<SIZE_OF_ARRAY(results_data); i++){
            if(results_data[i] == i){
                cout << "True" << endl;
            }else{
                is_hsj = false;
            }
        }

        if(is_hsj){
            cout << Guidance << endl;
            cout << LINE << endl;

            for (size_t i = 0; i < SIZE_OF_ARRAY(results_data); ++i) {
                results_data[i] = -1;
                cout << results_data[i] << endl;
            }

            // ラムダ式を使用して配列の各要素に操作を適用
            // for_each(begin(results_data), end(results_data), [](int &n){ n *= 0; });

        }


        int num = 10;
        cout << num << endl;
        // num = apps::sensor::test(num);
        num = test(num);
        cout << num << endl;

        return is_hsj;

    }
  }
}

int main(void){
    
    
    std::vector<int> results_data = {{-1, -1, -1,     -1,-1}};
    //std::array<int, 3> x = {-1, -1. -1}; // C++14～
    auto size = results_data.size();  // 3
    cout << "x size : " << size << endl;
    //auto size = std::size(x);  // C++17～

    // cout << "size : " << sizeof(results_data) << endl; // 24byte
    // cout << "size : " << SIZE_OF_ARRAY(results_data) << endl; // 6byte

    bool judge;
    judge = apps::sensor::demo_func(results_data); // &resulus_data[0]
    if(judge) cout << "Guidance >> Judge == " << Guidance << " (2times)" << endl;

}
