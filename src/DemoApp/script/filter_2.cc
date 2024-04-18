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
#define THROUGH_QUEUE_SIZE 5 // 12
#include <deque>

// 0:reset, 1:push, 2:judge
#define RESET  0
#define PUSH   1
#define JUDGE  2

// 追加
std::deque<uint32_t> THROUGH_QUEUE;
#define THROUGH_QUEUE_SIZE_AFTER_GUIDANCE   40
#define THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE  20
#define NONE 0
#define JUDGE_AFTER_GUIDANCE  2
#define JUDGE_AFTER_NOTCHANGE  3

namespace DemoApp{
    bool ThroughCheck_filter(uint32_t action_result, uint32_t JUDGE_MODE, uint32_t JUDGE_PARAM){
    
        // 0:reset, 1:push, 2:judge

        int reset_value = -1;
        bool filter = false;
        
        // push
        if(JUDGE_MODE == JUDGE_AFTER_GUIDANCE){
            THROUGH_QUEUE.push_front(action_result);
            // THROUGH_QUEUE.pop_back();

            // for (int i = 0; i < THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE; ++i){
            //     cout << "  i:" << i << " >> COMBO QUEUE : " << THROUGH_QUEUE[i] << endl;
            // }
            
            std::cout << "Through Size: " << THROUGH_QUEUE.size() << std::endl;
            
            if(THROUGH_QUEUE.size() > JUDGE_PARAM){ // THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE){
                filter = true;
                THROUGH_QUEUE.clear();
                // sizeは0になるので、ThroughCheck_filter()では、なくてもいい
                for (int i = 0; i < THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE; ++i){
                    THROUGH_QUEUE.push_front(reset_value); // -1
                    THROUGH_QUEUE.pop_back();
                }
            }
        }
        if(JUDGE_MODE == JUDGE_AFTER_NOTCHANGE){
            THROUGH_QUEUE.push_front(action_result);
            // THROUGH_QUEUE.pop_back();
            // for (int i = 0; i < THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE; ++i){
            //     cout << "  i:" << i << " >> COMBO QUEUE : " << THROUGH_QUEUE[i] << endl;
            // }
            std::cout << "Through Size: " << THROUGH_QUEUE.size() << std::endl;
            
            if(THROUGH_QUEUE.size() > JUDGE_PARAM){ // THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE){
                filter = true;
                THROUGH_QUEUE.clear();
                // sizeは0になるので、ThroughCheck_filter()では、なくてもいい
                for (int i = 0; i < THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE; ++i){
                    THROUGH_QUEUE.push_front(reset_value); // -1
                    THROUGH_QUEUE.pop_back();
                }
            }
        }

        if(JUDGE_MODE == RESET){
            THROUGH_QUEUE.clear();
            // // sizeは0になるので、ThroughCheck_filter()では、なくてもいい
            // for (int i = 0; i < THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE; ++i){
            //     THROUGH_QUEUE.push_front(reset_value); // -1
            //     THROUGH_QUEUE.pop_back();
            // }
        }

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
    
    // bool judge = true;
    // int JUDGE_MODE = JUDGE;
    // int JUDGE_PARAM = THROUGH_QUEUE_SIZE_AFTER_GUIDANCE;

    for(int i=0; i<COMBO.size(); i++){
        cout << "num: " << i << endl;
        result = DemoApp::ThroughCheck_filter(COMBO[i], JUDGE_AFTER_GUIDANCE, THROUGH_QUEUE_SIZE_AFTER_GUIDANCE);

        if(result){
            cout << "  >> filter: true" << " RESET!!!!!" << endl;
            // break;
            cout << "**********************************" << endl;
        }
        else cout << "  >> filter: false" << endl;
    }

    // int JUDGE_MODE = RESET;
    // int JUDGE_PARAM = THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE;

    cout << "**********************************" << endl;
    cout << "**********************************" << endl;
    cout << "**********************************" << endl;
    cout << "  >> filter: true" << " RESET!!!!!" << endl;
    DemoApp::ThroughCheck_filter(NONE, RESET, NONE);

    for(int i=0; i<COMBO.size(); i++){
        cout << "num: " << i << endl;
        result = DemoApp::ThroughCheck_filter(COMBO[i], JUDGE_AFTER_NOTCHANGE, THROUGH_QUEUE_SIZE_AFTER_NOTCHANGE);

        if(result){
            cout << "  >> filter: true" << " RESET!!!!!" << endl;
            // break;
            cout << "**********************************" << endl;
        }
        else cout << "  >> filter: false" << endl;
    }

    // result = DemoApp::ThroughCheck_filter(COMBO[i], RESET, );
}