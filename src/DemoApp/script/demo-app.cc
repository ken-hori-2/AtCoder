#include <bits/stdc++.h>
using namespace std;

int main() {


    // string S;
    // cin >> S;

    // // ここにプログラムを追記
  
    // //   int i, result = 1;
    // //   for(i = 0; i < S.size(); i++){
    // //     if(S.at(i) == '+'){
    // //         result += 1;
    // //     }else if(S.at(i) == '-'){
    // //         result -= 1;
    // //     }
    // //   }

    // //   cout << result << endl;
    // cout << S << endl;

    int *p;

    p = new int[3];
    cout << *p << endl; // 未定義の動作

    cout << sizeof(*p) << endl; // アドレス=8byte　指す先はint=4byte

    for(int i=0; i<3; i++){
        p[i] = i;
        // cout << *p << endl;
        cout << p[i] << endl;
    }
    // for(int i=0; i<3; p++, i++){
    //     *p = i;
    //     cout << *p << endl;
    // }
    // このままだとpの指す先が変わったまま


    // p = new int[3];
    // cout << *p << endl; // 未定義の動作

    cout << "*****" << endl;

    bool is_hsj = true;
    
    for(int i=0; i<3; p++, i++){
        // *p = i;
        // cout << *p << endl;
        // cout << p[i] << endl;
        if(*p == i){
            cout << "True" << endl;
        }else{
            is_hsj = false;
        }
    }
    // p--;
    // p--;
    // p--;
    p-=3;

    if(is_hsj) cout << "hop step jump !" << endl;

    
    if(p != nullptr){
        free(p);
        cout << "*****" << endl;
    }
    // cout << *p << endl;
    // p.clear();
    // cout << *p << endl;
    for(int i=0; i<3; p++, i++){
        // *p = i;
        cout << *p << endl;
    }

}
