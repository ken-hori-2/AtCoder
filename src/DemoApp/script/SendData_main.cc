#include <bits/stdc++.h>
using namespace std;

#define APPNAME "test_func"
#include <cstdlib>
#include <iostream>
#include <string>
#define SEND_SIZE args.size()
#define TAP_CLASSIFICATION_STABLE 0
#define NUMBER_OF_DATA 3

// void SendData(const std::vector<std::string>& args); // argはベクター型

// std::string GetStatusStr(netcom::status_t status) {
//   std::string s;
//   switch (status) {
//     #define STATUS_STR(status) case netcom::status_t::status: s = #status; break
//         STATUS_STR(kSuccess);
//         STATUS_STR(kSuccessNotSwitched);
//         STATUS_STR(kInvalidParam);
//         STATUS_STR(kInvalidConnectionStopped);
//         STATUS_STR(kInvalidNoConnecting);
//         STATUS_STR(kInvalidNotSupport);
//         STATUS_STR(kInvalidNoDeviceInfo);
//         STATUS_STR(kInvalidBufferFull);
//         STATUS_STR(kInvalidQueueFull);
//         STATUS_STR(kInvalidState);
//         STATUS_STR(kTimeout);
//     #undef STATUS_STR
//   }
//   return s;
// }

uint8_t StrToUint8(const std::string& str) {
  return static_cast<uint8_t>(std::stoul(str));
}
uint16_t StrToUint16(const std::string& str) {
  return static_cast<uint16_t>(std::stoul(str));
}

int main(){

    
    std::vector<std::string> args = {{"1", "2", "1"}}; // id, data, size // 送信したいデータ ... これをsend_dataにpush_backして格納する（今はargsをそのまま渡してもいいかも）

    uint8_t connection_id; // 8はうまく出力されない // uint16_t connection_id; // 16は出力される
    uint16_t size;
    // std::vector<uint8_t> send_data; // ここがuint8_tだとうまく中身が表示されない ... 符号なしintだからただ表示がされていないだけ？
    // std::vector<uint8_t> send_data = {{1, 2, 3}};

    connection_id = StrToUint8(args.at(1)); // connection_id = static_cast<uint8_t>(std::stoul(args.at(1)));
    size = StrToUint16(args.at(2)); // size = static_cast<uint16_t>(std::stoul(args.at(2)));
                
    // auto& client = GetManager<NetworkCommunicationClient>();
    // if (args.size() <= 2) {
    //     NetworkCommunicationHelp();
    //     return;
    // }
    
    // push_backしないとデータが確保されない
    // しないなら以下のように最初から格納する
    std::vector<uint8_t> send_data = {{1, 2, 3}};

    
    // // for (size_t i = 3; args.size() > i; i++) { // i=3から始まってサイズより大きくなるまでループを回す
    // for (size_t i = 0; i < args.size(); i++) { // i=3から始まってサイズより大きくなるまでループを回す
    //     std::cout << "i:" << i << std::endl;
    //     send_data.push_back(StrToUint8(args.at(i))); // キューの初期化
    // }
    // send_data.push_back(12);

    // for(int i=0; i<NUMBER_OF_DATA; i++){ // 初期化
    //     send_data[i] = 0; // -1; // uint8_t だと符号がないので、-1 は 255になる
    // }
    send_data[0] = 1;
    send_data[1] = 2;
    send_data[2] = 1;

    

    // std::cout << "send_data.data():" << send_data[0] << std::endl;
    // std::cout << "send_data.data():" << send_data[1] << std::endl;
    // std::cout << "send_data.data():" << send_data[2] << std::endl;
    // std::cout << "send_data.data():" << send_data.data() << " size:" << size << " id:" << connection_id << std::endl; 
    printf("test_1:  %p\n", send_data.data());
    printf("test_2:  %d\n", *send_data.data());
    printf("test_3-1:%d\n", send_data.data()[0]);
    printf("test_3-2:%d\n", send_data.data()[1]);
    printf("test_3-3:%d\n", send_data.data()[2]);
    printf("test_3-3:%d\n", send_data.data()[3]);
    printf("test_3-3:%d\n", send_data.data()[4]);

    // Send_Data_test();

    // const auto status = client.SendData(connection_id, send_data.data(), size); // statusにデータ送信が成功したかの判定を格納
    // // TWSS_RINFO(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, GetStatusStr(status), connection_id);
    // std::cout(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, status, connection_id);

}