#include <bits/stdc++.h>
using namespace std;

#define APPNAME "test_func"
#include <cstdlib>
#include <iostream>
#include <string>
#define SEND_SIZE args.size()
#define TAP_CLASSIFICATION_STABLE 0

// void SendData(const std::vector<std::string>& args); // argはベクター型

// uint8_t StrToUint8(const std::string& str) {
//   return static_cast<uint8_t>(std::stoul(str));
// }
// uint16_t StrToUint16(const std::string& str) {
//   return static_cast<uint16_t>(std::stoul(str));
// }
int StrToUint8(const std::string& str) {
  return static_cast<int>(std::stoul(str));
}
int StrToUint16(const std::string& str) {
  return static_cast<int>(std::stoul(str));
}

int main(){

    std::vector<std::string> args = {{"000", "111", "222", "333"}}; // {"stable", "walking", "runnning"};
    // std::vector<std::string> args_test = {{"0", "1", "2", "3"}};

    // uint8_t connection_id;
    int connection_id;
    // uint16_t size;
    int size;
    // std::vector<uint8_t> send_data_test;
    std::vector<int> send_data;

    // init queue
    // for (int i = 0; i < SEND_SIZE; ++i){
    //     send_data.push_front(TAP_CLASSIFICATION_STABLE);
    // }

    // std::cout << args.at(1) << std::endl;
    // std::cout << args.at(2) << std::endl;

    connection_id = StrToUint8(args.at(1));
    size = StrToUint16(args.at(2));         // = valueのこと??

    // GATT の中身 ... 
    // send data の中身はこうなっている? ... {???, conection_id, size, ???}
    
    // 以下ネットから
    // 0: ハンドル(Handle) - デバイス内で属性に連番で付与される16bitの番号
    // 1: 型(Type) - 値に何が入っているかを示すUUID。サービスや特性(Characteristic)を表す
    // 2: 値(Value) - 属性の値で長さはいろいろ
    // 3: 権限(Permission) - 読み書き等の権限を表す

    // const auto status = client.SendData(connection_id, send_data.data(), size); // (型, 値, サイズ) ???

    // *****
    // ちなみに実際に渡すSendData関数は以下
    
    /// 可変長データの送信を行う
    /// 実行タイプ：非同期型
    ///
    /// @param id
    /// [in] 接続ID(対向機/左右対向)
    /// @param data
    /// [in] 送信データの開始ポインタ
    /// @param size
    /// [in] 送信データサイズ
    /// @param observer
    /// [in] 通知を受けるオブザーバー

    // netcom::status_t SendData(uint8_t id, uint8_t* data, uint16_t size, INetworkCommunicationObserver* observer = nullptr);

    // inline netcom::status_t SendData(connection_id_t id, uint8_t* data, uint16_t size, INetworkCommunicationObserver* observer = nullptr){
    //     return(SendData(static_cast<uint8_t>(id),data,size,observer));
    // }

    // uint8_t id, uint8_t* data, uint16_t size // この型にする
    // *****





    // connection_id = static_cast<uint8_t>(std::stoul(args.at(1)));
    // size = static_cast<uint16_t>(std::stoul(args.at(2)));

    // unsigned long x = std::stoul("10"); // std::stoul("10", nullptr, 10);
    // std::cout << std::stoul("10") << std::endl;
    // std::cout << "10" << std::endl;
    // auto x = static_cast<uint16_t>(std::stoul(args.at(1))); //"10"));
    // std::cout << x << std::endl;

    // auto& client = 100; // GetManager<NetworkCommunicationClient>();
    // if (args.size() <= 2) {
    //     NetworkCommunicationHelp();
    //     return;
    // }


    // std::cout << "data:" << send_data.data() << "size:" << size << "id:" << connection_id << std::endl; 
    
    // for (size_t i = 3; args.size() > i; i++) { // i=3から始まってサイズより大きくなるまでループを回す
    for (size_t i = 0; i < args.size(); i++) { // i=3から始まってサイズより大きくなるまでループを回す
        std::cout << "i:" << i << std::endl;
        send_data.push_back(StrToUint8(args.at(i))); // キューの初期化

        // send_data_test.push_back(StrToUint8(args_test.at(i))); // キューの初期化
    }
    // send_data.push_back(12);

    // std::cout << " data:" << send_data << std::endl;

    // const auto status = client.SendData(connection_id, send_data.data(), size); // statusにデータ送信が成功したかの判定を格納
    // // TWSS_RINFO(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, GetStatusStr(status), connection_id);
    // std::cout(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, status, connection_id);

    std::cout << "send_data.data():" << send_data[0] << " &:" << &send_data[0] << std::endl;
    std::cout << "send_data.data():" << send_data[1] << std::endl;
    std::cout << "send_data.data():" << send_data[2] << std::endl;

    // send_data.data() = 先頭アドレス
    std::cout << "send_data.data():" << send_data.data() << " size:" << size << " id:" << connection_id << std::endl; 
    // std::cout << "send_data_test.data():" << send_data_test.data() << " size:" << size << " id:" << connection_id << std::endl; 

    // Send_Data_test();
}