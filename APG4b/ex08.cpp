#include <bits/stdc++.h>
using namespace std;

int main() {
  int p;
  cin >> p;

//   // �p�^�[��1
//   int price;
//   if (p == 1) {
//     cin >> price;
//   }

//   // �p�^�[��2
//   string text;
//   if (p == 2) {
//     cin >> text >> price;
//   }

//   int N;
//   cin >> N;

//   if(p == 2){
//     cout << text << "!" << endl;
//   }
//   cout << price * N << endl;

  // ��L���Ə璷. �ȉ��͏����s�������点��.
  // �ł��������͌�ɂ����o�͂���Ă��܂��̂�, ����ł����̂��s��.
  
  if (p == 2) {
    string text;
    cin >> text;
    cout << text << "!" << endl;
  }

  int price, N;
  cin  >> price >> N;
  cout << price * N << endl;
}
