#include <bits/stdc++.h>
using namespace std;

int main() {
  int p;
  cin >> p;

//   // パターン1
//   int price;
//   if (p == 1) {
//     cin >> price;
//   }

//   // パターン2
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

  // 上記だと冗長. 以下は少し行数が減らせる.
  // でも文字入力後にすぐ出力されてしまうので, これでいいのか不明.
  
  if (p == 2) {
    string text;
    cin >> text;
    cout << text << "!" << endl;
  }

  int price, N;
  cin  >> price >> N;
  cout << price * N << endl;
}
