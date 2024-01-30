#include <bits/stdc++.h>
using namespace std;

int main() {
  int N, A;
  cin >> N >> A;

  // ここにプログラムを追記
  int i;
  for(i = 0; i<N; i++){
    string op;
    // char op; // これだとエラー
    int B;

    cin >> op >> B;

    // if(B){ // ここで判定してしまうと+, -, *は0でもいいのにできない
    if(op == "+"){
        A += B;
    }else if(op == "*"){
        A *= B;
    }else if(op == "-"){
        A -= B;
    }else if(op == "/" && B != 0){
        A /= B;
    }else{
        cout << "error" << endl;
        break;
    }

    cout << i+1 << ":" << A << endl;

    // }else{
    //     cout << "error" << endl;
    //     break;
    // }
  }
}
