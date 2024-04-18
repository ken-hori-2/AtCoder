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

#define NUMBER_OF_DATA 3
#define NUMBER_OF_ACTION_COMBO 6
#define COMBO_QUEUE_SIZE 5 // 12
#include <deque>
std::deque<uint32_t> COMBO_QUEUE;

namespace DemoApp{

    // std::deque<uint32_t> COMBO_QUEUE;

    bool Combo_filter(uint32_t classification, bool judge=false){ // }, float probability){
      // data push
        //   if (static_cast<double>(probability) < ACCURACY_THRESHOLD){
        //     classification = -1;
        //   }
        //   cout << "COMBO QUEUE : " << COMBO_QUEUE[-1] << endl;
        // for (size_t i = 0; i < COMBO_QUEUE.size(); ++i) {
        //         cout << COMBO_QUEUE.at(i) << "; ";
        //     }
        // cout << endl;
        
        // 先頭に値を追加
        // cout << "classification:" << classification << endl;
        // bool judge = true;
        COMBO_QUEUE.push_front(classification);
        
        cout << "size:" << COMBO_QUEUE.size() << endl; // 1
        
        for (size_t i = 0; i < COMBO_QUEUE.size(); ++i) {
                cout << " add >> i:" << i << ", queue:" << COMBO_QUEUE.at(i) << "; ";
            }
        cout << endl;

        // 末尾から値を削除
        COMBO_QUEUE.pop_back();

        // for (size_t i = 0; i < COMBO_QUEUE.size(); ++i) {
        //         cout << COMBO_QUEUE.at(i) << "; ";
        //     }
        // cout << endl;

        // filter処理
        int result = -1;
        bool filter = true;
        
        for (int i = 0; i < COMBO_QUEUE_SIZE; ++i){

            cout << "  i:" << i << " >> COMBO QUEUE : " << COMBO_QUEUE[i] << endl;

            if(COMBO_QUEUE[i] != classification && filter == true){
            filter = false;
            }
        }

        // if(filter){
        //     result = classification;
        // }
        if(filter|| judge){
            COMBO_QUEUE.clear();
            cout << "  >> Clear" << endl;
            // for (size_t i = 0; i < COMBO_QUEUE.size(); ++i) {
            //     cout << "   >> " << COMBO_QUEUE.at(i) << "; ";
            // }
            int reset_num = -1;
            for (int i = 0; i < COMBO_QUEUE_SIZE; ++i){
                COMBO_QUEUE.push_front(reset_num);
                COMBO_QUEUE.pop_back();
            }
        }

        // return result;
        return filter;
    }
}


int main(){

    bool result;
    // std::vector<int> COMBO = {0, 1, 1, 2, 1, 1, 1, 1, 2, 2};
    std::vector<int> COMBO = {0, 1, 1, 
                              2, 1, 1, 
                              1, 1, 2, 
                              2, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              0, 1, 1,
                              0, 2, 1,
                              
                              1, 2, 0,
                              1, 1, 1,
                              1, 2, 0,
                              1, 2, 1,
                              1, 1, 1,
                              2, 1, 0,
                              };
    
    bool judge = true;
    for(int i=0; i<COMBO.size(); i++){
        cout << "num: " << i << endl;
        if(i%3==0){
            // result = DemoApp::Combo_filter(COMBO[i], judge);
            result = DemoApp::Combo_filter(COMBO[i]);

            if(result) cout << "  >> filter: true" << " RESET!!!!!" << endl;
            else cout << "  >> filter: false" << endl;
        }else cout << " >> pass" << endl;
    }

    // if(result) cout << "true" << endl;
    // cout << "result:" << result << endl;
    // if(result) cout << "filter: true" << endl;
    // else cout << "false" << endl;
}