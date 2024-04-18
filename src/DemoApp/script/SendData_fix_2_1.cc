#include <bits/stdc++.h>
using namespace std;

#define ACTION_CLASSIFICATION_RUNNING 2


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


class GATT_DATA{

    private:
        std::string send_data;
        std::vector<uint8_t> send_data_fix;
        uint8_t connection_id;
        uint16_t size;
    public:
        // 初期化
        GATT_DATA(std::string data, uint8_t id, uint16_t data_size){
            send_data = data;
            connection_id = id;
            size = data_size;
        }

        uint8_t StrToUint8(const std::string& str) {
            // return static_cast<uint8_t>(std::stoul(str));
            return static_cast<std::vector<uint8_t>>(std::stoul(str));
        }
        // void StrToUint8(const std::string& str) {
        //     send_data_fix = static_cast<uint8_t>(std::stoul(str));
        // }
        // uint16_t StrToUint16(const std::string& str) {
        // return static_cast<uint16_t>(std::stoul(str));
        // }

    
        // 送信データ変更
        void SendData(std::vector<uint8_t> data_fix, uint8_t id, uint16_t data_size){
            std::cout << "Call GATT DATA !!" << std::endl;
            send_data_fix = data_fix;
            connection_id = id;
            size = data_size;
        }

        // std::vector<uint8_t> GetData(){
        uint8_t* GetData(){
            return send_data_fix.data();
        }

        // 戻り値をベクター型にするなら以下のようにする
        std::vector<uint8_t>* GetData_2(){
            return &send_data_fix; // .data(); // &send_dataはベクター全体の先頭アドレスとして認識されるが、.data()は先頭のuint8_tのアドレスと認識されるからエラーになるのかも
        }

};

int main(){

    int classification = 2; // 0;

    // std::vector<uint8_t> send_data;
    // std::vector<uint8_t> send_data(1); // = {100};
    std::vector<uint8_t> send_data_fix(1); // , ""); // send_data_fix(1, "");
    std::string send_data = "";

    uint8_t connection_id = 1;
    uint16_t size = 1;
    // SDK
    // auto& client = GetManager(); // easel::GetManager<pst::netcom::NetworkCommunicationClient>();
    GATT_DATA client(send_data, connection_id, size); // Class初期化

    if(classification == ACTION_CLASSIFICATION_RUNNING){
        // TWSS_RINFO(APPNAME"::%s dnnrt result RUNNING", __func__);
        // send_data = {{ACTION_CLASSIFICATION_RUNNING}};
        // send_data = {{ACTION_CLASSIFICATION_RUNNING}, 1, 0}; // 複数個送ることもできる
        send_data = {{"RUNNING"}}; // 試しに文字列
        
        std::cout << "Running Detection" << std::endl;
    }else{
        // TWSS_ERR(APPNAME"::%s Unknown classification received", __func__);
        // send_data = {{200}};
        send_data = {{"Error"}};
        std::cout << "Not Detection" << std::endl;
    }

    // std::cout << "send_data: " << *send_data.data() << std::endl;
    // printf("send_data: %d\n", *send_data.data());
    // SDK
    // const auto status = client.SendData(connection_id, send_data.data(), size); // statusにデータ送信が成功したかの判定を格納
    // TWSS_RINFO(APPNAME"::%s status[%s] connection_id[%d], *data[%d]", __func__, status, connection_id, *send_data.data());




    // client.StrToUint8(send_data);

    client.SendData(client.StrToUint8(send_data), connection_id, size); // Class内のデータ書き換え
    std::cout << "status: " << *client.GetData() << std::endl;
    printf("send_data: %d\n", *client.GetData());

    uint8_t* p = client.GetData();
    printf("send_data: %d\n", p[1]); // p->at(1));
    printf("send_data: %d\n", p[2]); // p->at(2));

    std::cout << "*****" << std::endl;
    std::vector<uint8_t>* vec_p = client.GetData_2();
    printf("send_data: %d\n", vec_p->at(1)); // at()はベクターで使える
    printf("send_data: %d\n", vec_p->at(2));
}