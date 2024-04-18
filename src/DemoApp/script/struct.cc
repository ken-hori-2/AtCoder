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

# define NUMBER_OF_ACTION_COMBO 6
struct ComboPattern {
        int idx;
        string ComboName;
        std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
};
// vector<ComboPattern> 
ComboPattern comp_arr[] = { // idx, Name, Value
                            {0, "Combo-A", {0, 1, 2}},
                            {1, "Combo-B", {0, 2, 1}},
                            {2, "Combo-C", {1, 0, 2}},
                            {3, "Combo-D", {1, 2, 0}},
                            {4, "Combo-E", {2, 0, 1}},
                            {5, "Combo-F", {2, 1, 0}},
                            // {3, "Combo-D", {1, 2, 0}},
                            };

int findIndex2(std::vector<int> arr);
bool judge_func(std::vector<int> arr, std::vector<int> data, string comp_arr, auto CP[]);
int find_func(std::vector<int> arr, int item);

int main() {
    // int n;
    // cout << "Input num : " ;
    // cin >> n;
    // vector<int> one_case;
    // for (int i = 0; i < n; i++) { // one_case = {0,1,2,3,...n-1} とする
    //     one_case.emplace_back(i);
    // }

    // cout << one_case << endl;

    // do {
    //     for (auto num : one_case) {
    //         cout << num << " ";
    //     }
    //     cout << "\n";

    // } 
    // while (next_permutation(one_case.begin(), one_case.end()));
    // // 順列の最後になるまで one_case を並び替えながらループ

    // struct ComboPattern {
    //     // int Act1st;
    //     // int Act2nd;
    //     // int Act3rd;
    //     string ComboName;

    //     // std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
    //     // int Act1st;
    //     // int Act2nd;
    //     // int Act3rd;
    //     std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
    // };

    // struct ComboPattern CP1 = {
    //                           "Combo A",
    //                           {0, 1, 2}
    //                          };
    // struct ComboPattern CP2 = {
    //                           "Combo B",
    //                           {2, 1, 0}
    //                          };                             

    // 構造体初期化
    // memset(&CP,
    //        0x00,
    //        sizeof(CP) );

    // CP.ComboName = "TestAct1";
    // CP.COMBO = {2, 1, 0};
    // CP.COMBO = {
    //             "A",
    //             {2, 1, 0}
    //             };
    // cout << CP.COMBO << endl;
    // for(std::vector<int>::iterator i=begin(CP.COMBO); i!=end(CP.COMBO); ++i ) {
    //   std::cout << i << std::endl;
    // }


    // cout << "Name:" << CP2.ComboName << endl;
    // // for (auto const& value : CP2.COMBO){
    // for (auto value : CP2.COMBO){
    //     cout << value << " ";
    // }
    // cout << endl;




    // int n;
    // cout << "Input num : " ;
    // cin >> n;
    // vector<int> one_case;
    // for (int i = 0; i < n; i++) { // one_case = {0,1,2,3,...n-1} とする
    //     one_case.emplace_back(i);
    // }
    // do {
    //     for (auto num : one_case) {
    //         cout << num << " ";
    //     }
    //     cout << "\n";

    // } 
    // while (next_permutation(one_case.begin(), one_case.end()));
    // 順列の最後になるまで one_case を並び替えながらループ







    // struct ComboPattern {
    //     string ComboName;
    //     std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
    // };
    // // vector<ComboPattern> 
    // ComboPattern comp_arr[] = {
    //                             {"Combo-A", {0, 1, 2}},
    //                             {"Combo-B", {0, 2, 1}},
    //                             {"Combo-C", {1, 0, 2}},
    //                             {"Combo-D", {1, 2, 0}},
    //                             {"Combo-E", {2, 0, 1}},
    //                             {"Combo-F", {2, 1, 0}},
    //                             };

    // for (const auto &arr : comp_arr) {
    //     // cout << "Name: " << arr.name << endl
    //     //     << "CEO: " << arr.ceo << endl
    //     //     << "Income: " << arr.income << endl
    //     //     << "Employees: " << arr.employess << endl
    //     //     << endl;
    //     cout << arr << endl;
    // }
    // for (auto value : comp_arr){
    //     cout << value << " ";
    // }
    // cout << comp_arr[0] << endl;
    // cout << comp_arr[1] << endl;
    
    // for (auto const& value : CP2.COMBO){
    for (auto value : comp_arr){
        cout << "Name: " 
             << value.ComboName 
             << endl;
        cout << "Combo: ";
        for (auto act : value.COMBO){
            // cout << "value:" << value << " ";
            cout << act 
                 << " ";
        }
        cout << endl;
    }

    auto pos2 = findIndex2(comp_arr->COMBO);
    
    if(pos2 != -100){
        cout << "Found the element " << 10 << " at position " << pos2 << endl;
    }else{
        cout << "Could not found the element " << 10 << " in vector" << endl;
    }
    cout << typeid(pos2).name() << endl; // pos2 = int




    // vector<int> results_data = {{1, 0, 2}}; // 検索対象のデータ
    vector<int> results_data = {{2, 0, 1}}; // 検索対象のデータ
    
    bool judge = judge_func(comp_arr->COMBO, results_data, comp_arr->ComboName, comp_arr);
    // cout << "judge:" << judge << endl;
    if(judge) cout << "judge: true" << endl;
    
    return 0;
}




int findIndex2(std::vector<int> arr) {
    int item = 0; // -1; // 10;
    auto ret = std::find(arr.begin(), arr.end(), item);

    if (ret != arr.end()) return ret - arr.begin();
    // return -1;
    return -100;
}

bool judge_func(std::vector<int> arr, vector<int> data, string comp_arr, auto CP[]){

    vector<int> results_data = data; // {{1, 0, 2}};
    bool judge = false;
    int num;

    // if(results_data[i] == COMBO_012[i]){ // i){ // r[0]=0, r[1]=1, r[2]=2 ならtrue
    for(num=0; num<NUMBER_OF_ACTION_COMBO; num++){
        // for(int i=0; i<3; i++){ // for (auto value : comp_arr){
        //     // if(results_data[i] == arr[i]){
        //     if(results_data == CP[num].COMBO){
        //     // judge_012 = true;
        //         judge = true;
                
        //     // TWSS_RINFO(APPNAME"::%s ***************** results out %d !!!!!!!!!!> *****", __func__, results_data[i]);
        //     }else{
        //     // judge_012 = false;
        //         judge = false;
        //     }
        // }
        if(results_data == CP[num].COMBO){
            judge = true;
            cout << "num:" << num << endl;
            break;
        }
    }

    cout << "judge:" << judge << endl;
    // cout << "data:" << data[0] << endl;
    // cout << "arr:" << arr[0] << endl;

    if(judge){
        // cout << "TEST:" << comp_arr[i] << endl;
        // cout << "Name:" << comp_arr << endl;

        // for(int i=0; i<NUMBER_OF_ACTION_COMBO; i++){
            // cout << "results_data:" << results_data[i] << endl;
            // cout << "arr:" << arr[i] << endl;
            cout << "num:" << num << endl;
            cout << "Name:" << CP[num].ComboName << endl;

            // string Name = CP[i].ComboName;
            // cout << "Name:" << Name << endl;
            // switch (Name){
            cout << "idx:" << CP[num].idx << endl;
            
            // case "Combo-C":
            cout << " ********** " << endl;
            switch (CP[num].idx){
            case 0:
                cout << "num:" << num << endl;
                cout << "Success !!" << endl;
                cout << "Name:" << CP[num].ComboName << endl;
                break;
            case 2:
                cout << "num:" << num << endl;
                cout << "Success !!" << endl;
                cout << "Name:" << CP[num].ComboName << endl;
                break;
            
            default:
                // cout << "Success !!" << endl;
                cout << "num:" << num << endl;
                cout << "Name:" << CP[num].ComboName << endl;
                break;
                // break;
            }
        // }
    }

    return judge;
}

// int find_func(std::vector<int> arr, int item) {
//     int item = 0; // item;
//     auto ret = std::find(arr.begin(), arr.end(), item);

//     if (ret != arr.end()) return ret - arr.begin();
//     // return -1;
//     return -100;
// }