#include <bits/stdc++.h>
using namespace std;

#define ACTION_CLASSIFICATION_RUNNING 2
#define NUMBER_OF_ACTION 3



// Class内で別のClassを記述するパターン


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
class ComboPattern {
    private:
        std::string Name;
        std::vector<uint8_t> Data;
    public:
        ComboPattern(std::string InputName, std::vector<uint8_t> InputData){
            Name = InputName;
            Data = InputData;
        }
        std::string GetName(){
            return Name;
        }
        std::vector<uint8_t> GetData(){
            return Data;
        }
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

        void ChangeData_StrToUint8(std::string str, uint8_t id, uint16_t data_size){ // ,     ComboPattern* comp_arr){
        // void ChangeData_StrToUint8(std::string str, uint8_t id, uint16_t data_size,     ComboPattern& comp_arr){

            // 初期化
            ComboPattern comp_arr[] = {
                {"Stable",  {0}},
                {"Walking", {1}},
                {"Running", {2}},
                {"Other", {200}},
            };

            // std::string Name, Data;
            std::string Name;
            std::vector<uint8_t> Data;
            // Name = comp_arr->GetName();
            // Data = comp_arr->GetData();

            for(int num=0; num<NUMBER_OF_ACTION; num++){
            
                
                Name = comp_arr[num].GetName();
                Data = comp_arr[num].GetData();
                // auto test = comp_arr[num];
                
                if(str == Name){ // [num]){ // comp_arr[num].Name){ // コンボリストの中に一致するコンボはあるか
                    send_data = Data; // [num]; // comp_arr[num].Data;
                // if(str == test.GetName()){
                //     send_data = test.GetData();
                    
                    std::cout << "before: " << str; //  << std::endl;
                    
                    printf(" -> after: %d\n", *send_data.data());
                    
                    break;
                }
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
    std::vector<uint8_t> send_data(1, 100);
    uint8_t connection_id = 1;
    uint16_t size = 1;
    std::string str = "";

    // Class初期化
    // ComboPattern comp_arr[] = {
    //     {"Stable",  {0}},
    //     {"Walking", {1}},
    //     {"Running", {2}},
    // };
    GATT_DATA client(send_data, connection_id, size);

    if(classification == ACTION_CLASSIFICATION_RUNNING){
        // // TWSS_RINFO(APPNAME"::%s dnnrt result RUNNING", __func__);
        // // send_data = {{ACTION_CLASSIFICATION_RUNNING}};
        // send_data = {{ACTION_CLASSIFICATION_RUNNING}, 1, 0}; // 複数個送ることもできる
        // // send_data = {{"RUNNING"}}; // 試しに文字列
        str = "Running"; // 試しに文字列
        
        std::cout << "Running Detection" << std::endl;
    }else{
        // // TWSS_ERR(APPNAME"::%s Unknown classification received", __func__);
        // send_data = {{200}};
        str = "Other";
        std::cout << "Not Detection" << std::endl;
    }

    // std::cout << "send_data: " << *send_data.data() << std::endl; // うまく表示されない
    printf("Init Data > send_data: %d\n", *send_data.data());


    
    
    
    
    
    
    
    
    // どっちかを実行
    client.ChangeData_StrToUint8(str, connection_id, size); // ,     comp_arr);
    
    // どっちかを実行
    // client.SendData(send_data, connection_id, size); // Class内のデータ書き換え

    




    std::cout << "*****" << std::endl;
    
    // std::cout << "status: " << *client.GetData() << std::endl; // うまく表示されない
    printf("Init Data > send_data: %d\n", *client.GetData());

    // uint8_t* p = client.GetData();
    // printf("send_data: %d\n", p[1]); // p->at(1));
    // printf("send_data: %d\n", p[2]); // p->at(2));

    // std::cout << "*****" << std::endl;
    // std::vector<uint8_t>* vec_p = client.GetData_vector();
    // printf("send_data: %d\n", vec_p->at(1)); // at()はベクターで使える
    // printf("send_data: %d\n", vec_p->at(2));
}