#include <bits/stdc++.h>
using namespace std;

int main() {
  int N, A;
  cin >> N >> A;

  // �����Ƀv���O������ǋL
  int i;
  for(i = 0; i<N; i++){
    string op;
    // char op; // ���ꂾ�ƃG���[
    int B;

    cin >> op >> B;

    // if(B){ // �����Ŕ��肵�Ă��܂���+, -, *��0�ł������̂ɂł��Ȃ�
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
