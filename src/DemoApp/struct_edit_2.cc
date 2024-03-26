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

# define NUMBER_OF_DATA 3
# define NUMBER_OF_ACTION_COMBO 6





// bool judge_func(std::vector<int> data, auto *CP); // CP[]
// int findIndex2(std::vector<int> arr);
// int find_func(std::vector<int> arr, int item);


namespace apps {
  namespace demo {

    struct ComboPattern {
            int idx;
            string ComboName;
            std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
    };
    // vector<ComboPattern> 
    // ComboPattern comp_arr[] = { // idx, Name, Value
    //                             {0, "Combo-A", {0, 1, 2}},
    //                             {1, "Combo-B", {0, 2, 1}},
    //                             {2, "Combo-C", {1, 0, 2}},
    //                             {3, "Combo-D", {1, 2, 0}},
    //                             {4, "Combo-E", {2, 0, 1}},
    //                             {5, "Combo-F", {2, 1, 0}},
    //                             // {3, "Combo-D", {1, 2, 0}},
    //                             };
    ComboPattern CP[] = { // idx, Name, Value
                                {0, "Combo-A", {0, 1, 2}},
                                {1, "Combo-B", {0, 2, 1}},
                                {2, "Combo-C", {1, 0, 2}},
                                {3, "Combo-D", {1, 2, 0}},
                                {4, "Combo-E", {2, 0, 1}},
                                {5, "Combo-F", {2, 1, 0}},
                                // {3, "Combo-D", {1, 2, 0}},
                                };
    

    vector<int> results_data = {{0, 2, 1}}; // 検索対象のデータ
    

    
    

    // bool judge_func(vector<int> data, apps::demo::ComboPattern *CP){ // CP[]
    bool combo_judge_func(){ // vector<int> data, ComboPattern *CP){ // CP[]

        // vector<int> results_data = data; // {{1, 0, 2}};
        bool judge = false;
        int num;

        for(num=0; num<NUMBER_OF_ACTION_COMBO; num++){
            
            if(results_data == CP[num].COMBO){
                judge = true;
                cout << "num:" 
                    << num 
                    << endl;
                // cout << results_data << endl;
                // cout << CP[num].COMBO << endl;
                break;
            }
        }

        cout << "judge:" 
            << judge 
            << endl;

        if(judge){
            cout << "num:" << num << endl;
            cout << "idx:" << CP[num].idx << endl;
            cout << " ********** " << endl;
            cout << "Name:" << CP[num].ComboName << endl;
            cout << "Success !!" << endl;
            // switch (CP[num].idx){
            switch (num){
                case 0:
                    cout << "process 0" << endl;
                    break;
                case 1:
                    cout << "process 1" << endl;
                    break;
                case 2:
                    cout << "process 2" << endl;
                    break;
                case 3:
                    cout << "process 3" << endl;
                    break;
                case 4:
                    cout << "process 4" << endl;
                    break;
                case 5:
                    cout << "process 5" << endl;
                    break;
                default:
                    cout << "error" << endl;
                    break;
            }
        }

        return judge;
    }

    void Boot() {

        // for (auto value : apps::demo::comp_arr){
        for (auto value : CP){
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

        // vector<int> results_data = {{1, 0, 2}}; // 検索対象のデータ
        // vector<int> results_data = {{2, 0, 1}}; // 検索対象のデータ
        // vector<int> results_data = {{0, 1, 2}}; // 検索対象のデータ

        // vector<int> results_data = {{0, 2, 1}}; // 検索対象のデータ
        
        bool judge = apps::demo::combo_judge_func(); // results_data, apps::demo::comp_arr);
        
        if(judge) cout << "judge: true" << endl;
        
        // return 0;
    }
  }
}

    int main() {

        // for (auto value : apps::demo::comp_arr){
        //     cout << "Name: " 
        //         << value.ComboName 
        //         << endl;
        //     cout << "Combo: ";
        //     for (auto act : value.COMBO){
        //         // cout << "value:" << value << " ";
        //         cout << act 
        //             << " ";
        //     }
        //     cout << endl;
        // }

        // // vector<int> results_data = {{1, 0, 2}}; // 検索対象のデータ
        // // vector<int> results_data = {{2, 0, 1}}; // 検索対象のデータ
        // // vector<int> results_data = {{0, 1, 2}}; // 検索対象のデータ
        // vector<int> results_data = {{0, 2, 1}}; // 検索対象のデータ
        
        // bool judge = apps::demo::combo_judge_func(results_data, apps::demo::comp_arr);
        
        // if(judge) cout << "judge: true" << endl;
        apps::demo::Boot();
        
        return 0;
    }

    // bool judge_func(vector<int> data, auto *CP){ // CP[]

    //     vector<int> results_data = data; // {{1, 0, 2}};
    //     bool judge = false;
    //     int num;

    //     for(num=0; num<NUMBER_OF_ACTION_COMBO; num++){
            
    //         if(results_data == CP[num].COMBO){
    //             judge = true;
    //             cout << "num:" 
    //                 << num 
    //                 << endl;
    //             // cout << results_data << endl;
    //             // cout << CP[num].COMBO << endl;
    //             break;
    //         }
    //     }

    //     cout << "judge:" 
    //         << judge 
    //         << endl;

    //     if(judge){
    //         cout << "num:" << num << endl;
    //         cout << "idx:" << CP[num].idx << endl;
    //         cout << " ********** " << endl;
    //         cout << "Name:" << CP[num].ComboName << endl;
    //         cout << "Success !!" << endl;
    //         // switch (CP[num].idx){
    //         switch (num){
    //             case 0:
    //                 cout << "process 0" << endl;
    //                 break;
    //             case 1:
    //                 cout << "process 1" << endl;
    //                 break;
    //             case 2:
    //                 cout << "process 2" << endl;
    //                 break;
    //             case 3:
    //                 cout << "process 3" << endl;
    //                 break;
    //             case 4:
    //                 cout << "process 4" << endl;
    //                 break;
    //             case 5:
    //                 cout << "process 5" << endl;
    //                 break;
    //             default:
    //                 cout << "error" << endl;
    //                 break;
    //         }
    //     }

    //     return judge;
    // }

    // int findIndex2(std::vector<int> arr) {
    //     int item = 0; // -1; // 10;
    //     auto ret = std::find(arr.begin(), arr.end(), item);

    //     if (ret != arr.end()) return ret - arr.begin();
    //     // return -1;
    //     return -100;
    // }

    // int find_func(std::vector<int> arr, int item) {
    //     int item = 0; // item;
    //     auto ret = std::find(arr.begin(), arr.end(), item);

    //     if (ret != arr.end()) return ret - arr.begin();
    //     // return -1;
    //     return -100;
    // }