#include <bits/stdc++.h>
using namespace std;

#define ACTION_CLASSIFICATION_RUNNING 2
#define NUMBER_OF_ACTION 3


// // class GetManager{
// class SendData_fix{
//     private:
//         /* data */
//     public:
//         SendData_fix(/* args */);
//         ~SendData_fix();
// };

// SendData_fix::SendData_fix(/* args */){
// }

// SendData_fix::~SendData_fix(){
// }

struct ComboPattern {
        // int idx;
        // std::string ComboName;
        // std::vector<int> COMBO; //  = {Act1st, Act2nd, Act3rd};
        std::string Name;
        std::vector<uint8_t> Data;
};
// vector<ComboPattern> 
ComboPattern comp_arr[] = { // idx, Name, Value
                            // {0,  "Stable", {0, 1, 2}},
                            // {1,  "Walking", {0, 2, 1}},
                            // {2,  "Running", {1, 0, 2}},
                            {"Stable",  {0}},
                            {"Walking", {1}},
                            {"Running", {2}},
                            };


class GATT_DATA{

    private:
        std::vector<uint8_t> send_data;
        uint8_t connection_id;
        uint16_t size;
    public:
        // 初期化
        GATT_DATA(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){
            send_data = data;
            connection_id = id;
            size = data_size;
        }

        // 送信データ変更
        void SendData(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){
            std::cout << "Call GATT DATA !!" << std::endl;
            send_data = data;
            connection_id = id;
            size = data_size;
        }

        void ChangeData_StrToUint8(std::string str, uint8_t id, uint16_t data_size){

            for(int num=0; num<NUMBER_OF_ACTION; num++){
            
                if(str == comp_arr[num].Name){ // コンボリストの中に一致するコンボはあるか
                    send_data = comp_arr[num].Data;
                    // std::cout << "ChangeData_StrToUint8: " << send_data << std::endl;
                    std::cout << "before: " << str; //  << std::endl;
                    // std::cout << "after: " << send_data.at(0) << std::endl;
                    printf(" -> after: %d\n", *send_data.data());
                    // printf("ChangeData_StrToUint8 send_data: [before: %s after: %d]\n", str, *send_data.data());
                    break;
                }
                // else{
                //     std::cout << "Error! ChangeData_StrToUint8" << std::endl;
                // }
            }

            connection_id = id;
            size = data_size;

        }

        // std::vector<uint8_t> GetData(){
        uint8_t* GetData(){
            return send_data.data();
        }

        // 戻り値をベクター型にするなら以下のようにする
        std::vector<uint8_t>* GetData_vector(){
            return &send_data; // .data(); // &send_dataはベクター全体の先頭アドレスとして認識されるが、.data()は先頭のuint8_tのアドレスと認識されるからエラーになるのかも
        }

};

int main(){

    int classification = 2; // 0;

    // std::vector<uint8_t> send_data;
    std::vector<uint8_t> send_data(1); // = {100};
    // std::vector<uint8_t> send_data_fix(1, "");

    std::string str = "";
    
    uint8_t connection_id = 1;
    uint16_t size = 1;

    uint8_t send_data_num;

    GATT_DATA client(send_data, connection_id, size); // Class初期化

    if(classification == ACTION_CLASSIFICATION_RUNNING){
        // TWSS_RINFO(APPNAME"::%s dnnrt result RUNNING", __func__);
        // send_data = {{ACTION_CLASSIFICATION_RUNNING}};
        send_data = {{ACTION_CLASSIFICATION_RUNNING}, 1, 0}; // 複数個送ることもできる
        // send_data = {{"RUNNING"}}; // 試しに文字列
        str = "Running"; // 試しに文字列
        
        std::cout << "Running Detection" << std::endl;
    }else{
        // TWSS_ERR(APPNAME"::%s Unknown classification received", __func__);
        send_data = {{200}};
        std::cout << "Not Detection" << std::endl;
    }

    std::cout << "send_data: " << *send_data.data() << std::endl;
    printf("send_data: %d\n", *send_data.data());


    // どっちかを実行
    client.ChangeData_StrToUint8(str, connection_id, size);
    
    // どっちかを実行
    // client.SendData(send_data, connection_id, size); // Class内のデータ書き換え
    // // client.SendData(send_data_num, connection_id, size); // Class内のデータ書き換え


    std::cout << "status: " << *client.GetData() << std::endl;
    printf("send_data: %d\n", *client.GetData());

    // uint8_t* p = client.GetData();
    // printf("send_data: %d\n", p[1]); // p->at(1));
    // printf("send_data: %d\n", p[2]); // p->at(2));

    // std::cout << "*****" << std::endl;
    // std::vector<uint8_t>* vec_p = client.GetData_vector();
    // printf("send_data: %d\n", vec_p->at(1)); // at()はベクターで使える
    // printf("send_data: %d\n", vec_p->at(2));
}