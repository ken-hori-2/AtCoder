#include <bits/stdc++.h>
using namespace std;

#define ACTION_CLASSIFICATION_RUNNING 2
#define NUMBER_OF_ACTION 3

// main文内でClassを2つ実体化するパターン



class ConvertDetectionResultToUint8 {
    private:
        std::string Name;
        std::vector<uint8_t> Data;
    public:
        ConvertDetectionResultToUint8(std::string InputName, std::vector<uint8_t> InputData);
        std::string GetName();
        std::vector<uint8_t> GetData();
        ~ConvertDetectionResultToUint8();
};
// 初期化
ConvertDetectionResultToUint8::ConvertDetectionResultToUint8(std::string InputName, std::vector<uint8_t> InputData){ // コントラクタ
    Name = InputName;
    Data = InputData;
}
std::string ConvertDetectionResultToUint8::GetName(){
    return Name; // this->Name;
}
std::vector<uint8_t> ConvertDetectionResultToUint8::GetData(){
    return Data; // this->Data;
}
ConvertDetectionResultToUint8::~ConvertDetectionResultToUint8(){ // デストラクタ
    // 必要な場合は処理を記述する
}

class GATT_DATA{
    private:
        std::vector<uint8_t> send_data;
        uint8_t connection_id;
        uint16_t size;
    public:
        GATT_DATA(std::vector<uint8_t> data, uint8_t id, uint16_t data_size);
        void SendData_StrToUint8(std::string str, uint8_t id, uint16_t data_size,     ConvertDetectionResultToUint8* convert_table);
        uint8_t* GetData();
        std::vector<uint8_t>* GetData_vector();
        ~GATT_DATA();
};
// 初期化
// GATT_DATA(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){
GATT_DATA::GATT_DATA(std::vector<uint8_t> data, uint8_t id, uint16_t data_size){
    send_data = data;
    connection_id = id;
    size = data_size;
}

// 送信データ変更
void GATT_DATA::SendData_StrToUint8(std::string str, uint8_t id, uint16_t data_size,     ConvertDetectionResultToUint8* convert_table){ // ConvertDetectionResultToUint8& convert_table){
    std::string Name;
    std::vector<uint8_t> Data;
    for(int num=0; num<NUMBER_OF_ACTION; num++){
        Name = convert_table[num].GetName();
        Data = convert_table[num].GetData();
        auto test = convert_table[num];
        
        if(str == Name){ // コンボリストの中に一致するコンボはあるか
            send_data = Data;
            std::cout << "before: " << str;
            printf(" -> after: %d\n", *send_data.data());
            break;
        }
    }

    connection_id = id;
    size = data_size;

}
// std::vector<uint8_t> GetData(){
uint8_t* GATT_DATA::GetData(){
    return send_data.data();
}
// 戻り値をベクター型にするなら以下のようにする
std::vector<uint8_t>* GATT_DATA::GetData_vector(){
    return &send_data; // .data(); // &send_dataはベクター全体の先頭アドレスとして認識されるが、.data()は先頭のuint8_tのアドレスと認識されるからエラーになるのかも
}
GATT_DATA::~GATT_DATA(){
    // 必要な場合はここに処理する
}



int main(){

    int classification = 2; // 0;

    std::vector<uint8_t> send_data(1, 100); // 初期化時にデータを格納しないとうまく動かない
    // std::vector<uint8_t> send_data;
    uint8_t connection_id = 1;
    uint16_t size = 1;
    std::string str = "";

    // Class初期化
    ConvertDetectionResultToUint8 convert_table[] = {
        {"Stable",  {0}},
        {"Walking", {1}},
        {"Running", {2}},
        {"Other", {200}},
    };
    // GATT_DATA gatt_dataset; // send_data, connection_id, size);
    GATT_DATA gatt_dataset(send_data, connection_id, size);

    if(classification == ACTION_CLASSIFICATION_RUNNING){
        // // TWSS_RINFO(APPNAME"::%s dnnrt result RUNNING", __func__);
        str = "Running"; // 試しに文字列
        std::cout << "Running Detection" << std::endl;
    }else{
        // // TWSS_ERR(APPNAME"::%s Unknown classification received", __func__);
        str = "Other";
        std::cout << "Not Detection" << std::endl;
    }

    // std::cout << "Init Data > send_data: " << *send_data.data() << std::endl; // うまく表示されない
    printf("Init Data > send_data: %d\n", *send_data.data());


    // "**********"
    // 検出結果にデータ書き換え
    gatt_dataset.SendData_StrToUint8(str, connection_id, size,     convert_table);
    // "**********"


    std::cout << "*****" << std::endl;
    // std::cout << "Result Data > status: " << *gatt_dataset.GetData() << std::endl; // うまく表示されない
    printf("Result Data > send_data: %d\n", *gatt_dataset.GetData());

    // uint8_t* p = gatt_dataset.GetData();
    // printf("send_data: %d\n", p[1]); // p->at(1));
    // printf("send_data: %d\n", p[2]); // p->at(2));

    // std::cout << "*****" << std::endl;
    // std::vector<uint8_t>* vec_p = gatt_dataset.GetData_vector();
    // printf("send_data: %d\n", vec_p->at(1)); // at()はベクターで使える
    // printf("send_data: %d\n", vec_p->at(2));
}