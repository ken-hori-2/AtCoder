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
        std::vector<uint8_t> send_data;
        uint8_t connection_id;
        uint16_t size;

        bool judge;
    public:
        GATT_DATA(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){
            // std::vector<uint8_t> send_data = data;
            // uint8_t connection_id = id;
            // uint16_t size = data_size;
            send_data = data;
            connection_id = id;
            size = data_size;

            judge = false;
        }

    
        // 送信データ変更
        bool SendData(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){

            // std::cout << "GATT DATA: "<< GATT_DATA.send_data() << std::endl;
            std::cout << "Call GATT DATA !!" << std::endl;
            send_data = data;
            connection_id = id;
            size = data_size;

            judge = true;

            // return 10;
            return judge;

            // try{
            //     // std::cout << "GATT DATA: "<< GATT_DATA.send_data() << std::endl;
            //     std::cout << "Call GATT DATA !!" << std::endl;
            //     send_data = data;
            //     connection_id = id;
            //     size = data_size;

            //     throw "ERROR!!"; // "エラーが発生しました";
            // }catch(const char* msg){
            //     // std::cout << "例外が補足されました" << msg << std::endl;
            //     std::cout << "ERROR CATCH!!" << msg << std::endl;
            // }

        }

};

int main(){

    int classification = 0;

    // std::vector<uint8_t> send_data;
    std::vector<uint8_t> send_data(1); // = {100};
    uint8_t connection_id = 1;
    uint16_t size = 1;
    // SDK
    // auto& client = GetManager(); // easel::GetManager<pst::netcom::NetworkCommunicationClient>();
    GATT_DATA client(send_data, connection_id, size);

    if(classification == ACTION_CLASSIFICATION_RUNNING){
        // TWSS_RINFO(APPNAME"::%s dnnrt result RUNNING", __func__);
        send_data = {{ACTION_CLASSIFICATION_RUNNING}};

    }else{
        // TWSS_ERR(APPNAME"::%s Unknown classification received", __func__);
        send_data = {{200}};
    }

    // auto& client = GetManager(send_data, connection_id, size);
    // GATT_DATA client(send_data, connection_id, size);






    // SDK
    // const auto status = client.SendData(connection_id, send_data.data(), size); // statusにデータ送信が成功したかの判定を格納
    // TWSS_RINFO(APPNAME"::%s status[%s] connection_id[%d], *data[%d]", __func__, status, connection_id, *send_data.data());

    const auto status = client.SendData(send_data, connection_id, size); // statusにデータ送信が成功したかの判定を格納
    std::cout << "status: " << status << std::endl;
}