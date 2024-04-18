#include <bits/stdc++.h>
using namespace std;

#define APPNAME "test_func"
#include <cstdlib>
#include <iostream>
#include <string>
#define SEND_SIZE args.size()
#define TAP_CLASSIFICATION_STABLE 0

// void SendData(const std::vector<std::string>& args); // argはベクター型

uint8_t StrToUint8(const std::string& str) {
  return static_cast<uint8_t>(std::stoul(str));
}
uint16_t StrToUint16(const std::string& str) {
  return static_cast<uint16_t>(std::stoul(str));
}
// int StrToUint8(const std::string& str) {
//   return static_cast<int>(std::stoul(str));
// }
// int StrToUint16(const std::string& str) {
//   return static_cast<int>(std::stoul(str));
// }


namespace numerical_chars {
    inline std::ostream &operator<<(std::ostream &os, char c) {
        return std::is_signed<char>::value ? os << static_cast<int>(c)
                                        : os << static_cast<unsigned int>(c);
    }

    inline std::ostream &operator<<(std::ostream &os, signed char c) {
        return os << static_cast<int>(c);
    }

    inline std::ostream &operator<<(std::ostream &os, unsigned char c) {
        return os << static_cast<unsigned int>(c);
    }
}

int main(){

    std::vector<std::string> args = {{"000", "111", "222", "333"}}; // {"stable", "walking", "runnning"};
    std::vector<std::string> args_test = {{"10", "1", "2", "3"}};

    // uint8_t connection_id; // 8はうまく出力されないが
    uint16_t connection_id; // 16は出力される
    
    uint16_t size;
    std::vector<uint8_t> send_data_test; // 8は出力されないが
    // std::vector<uint16_t> send_data_test;   // 16は出力される
    std::vector<int> send_data; // ここがuint8_tだとうまく中身が表示されない ... 符号なしintだからただ表示がされていないだけ？

    // init queue
    // for (int i = 0; i < SEND_SIZE; ++i){
    //     send_data.push_front(TAP_CLASSIFICATION_STABLE);
    // }

    // std::cout << args.at(1) << std::endl;
    // std::cout << args.at(2) << std::endl;

    connection_id = StrToUint8(args.at(1));
    size = StrToUint16(args.at(2));
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

        send_data_test.push_back(StrToUint8(args_test.at(i))); // キューの初期化
    }
    // send_data.push_back(12);

    // std::cout << " data:" << send_data << std::endl;

    // const auto status = client.SendData(connection_id, send_data.data(), size); // statusにデータ送信が成功したかの判定を格納
    // // TWSS_RINFO(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, GetStatusStr(status), connection_id);
    // std::cout(APPNAME"::%s status[%s] connection_id[%d]\n", __func__, status, connection_id);

    std::cout << "send_data.data():" << send_data[0] << std::endl;
    std::cout << "send_data.data():" << send_data[1] << std::endl;
    std::cout << "send_data.data():" << send_data[2] << std::endl;

    std::cout << "send_data.data():" << send_data.data() << " size:" << size << " id:" << connection_id << std::endl; 
    std::cout << "send_data_test.data():" << send_data_test.data() << " size:" << size << " id:" << connection_id << std::endl; 
    printf("test_1:%p\n", send_data_test.data());
    printf("test_2:%d\n", *send_data_test.data());
    printf("test_3:%d\n", send_data_test.data()[0]);
    printf("test_3-2:%d\n", send_data_test.data()[1]);
    printf("test_3-3:%d\n", send_data_test.data()[2]);

    // Send_Data_test();


    uint8_t i = 100;
    using namespace numerical_chars;
    std::cout << i << std::endl; 
}