// #include <bits/stdc++.h>
// using namespace std;

#include <iostream>
  using std::cin;
  using std::cout;
  using std::endl;
  using std::string;

int main() {
  int A, B;
  string op;
  cin >> A >> op >> B;

//   if(!B){

    if (op == "+") {
        cout << A + B << endl;
    }
    // ‚±‚±‚ÉƒvƒƒOƒ‰ƒ€‚ð’Ç‹L
    else if(op == "-"){
        cout << A - B << endl;
    }else if(op == "*"){
        cout << A * B << endl;
    }else if(op == "/"){
        if(B){
            cout << A / B << endl;
        }else{
            cout << "error" << endl;
        }
    }else{
        cout << "error" << endl;
    }

//   }else{
//     cout << "error" << endl;
//   }

}
