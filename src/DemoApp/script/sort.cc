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

// #include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cout << "Input num : " ;
    cin >> n;
    vector<int> one_case;
    for (int i = 0; i < n; i++) { // one_case = {0,1,2,3,...n-1} とする
        one_case.emplace_back(i);
    }

    // cout << one_case << endl;

    // do {
    //     for (auto num : one_case) {
    //         cout << num << " ";
    //     }
    //     cout << "\n";

    // } 
    // while (next_permutation(one_case.begin(), one_case.end()));
    // 順列の最後になるまで one_case を並び替えながらループ






    std::vector<int> COMBO; //  = {{}};
    std::vector<std::vector<int>> vv = {{-1, -1, -1},
                                        {-1, -1, -1},
                                        {-1, -1, -1}
                                        };

    int i=0;
    for (auto num : one_case) {
            // cout << num << " ";
            // COMBO.insert(COMBO.end(), num);
            COMBO.push_back(num);
            vv[i].push_back(num);
            i++;
        }
        // cout << "\n";
        // cout << COMBO << endl;
    // for (auto num : COMBO) {
    //         cout << num << " ";
    //         // COMBO.insert(COMBO.end(), num);
    //     }
    //     cout << endl;
    while (next_permutation(one_case.begin(), one_case.end())){
        for (auto num : one_case) {
            // cout << num << " ";
            // COMBO.insert(COMBO.end(), num);
            COMBO.push_back(num);
            vv[i].push_back(num);
            i++;
        }
        // cout << "\n";
        // cout << COMBO << endl;
    }

    for (auto num : COMBO) {
            cout << num << " ";
            // COMBO.insert(COMBO.end(), num);
        }
        cout << endl;

    cout << COMBO.size() << endl;

    // for (auto num : vv) {
    //         cout << num << " ";
    //         // COMBO.insert(COMBO.end(), num);
    //     }
    //     cout << endl;
    for(int i=0; i<5; i++){
        cout << vv[i][0] << endl;
    }
    // cout << vv.size() << endl;




    // show(COMBO);
    
    // 指定した位置に要素を挿入
    // COMBO.insert(COMBO.begin() + 1, 10);
    // COMBO.insert(COMBO.end(), num);

    // show(COMBO);
    
    return 0;
}