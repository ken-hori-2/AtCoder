#include <bits/stdc++.h>
using namespace std;

int main() {
//   string str1, str2;
//   cin >> str1;
//   str2 = ", world!";

//   cout << str1 + str2 << endl; // これでもいい　str1 << str2
//   cout << (str1+str2).size() << endl; // 長さを測定
//   cout << (str1+str2).at(0) << endl; // strの先頭にアクセス
//   char a = '*'; // 一文字を扱うときは 'シングルコーテーション' で囲う ... ""は文字列として扱われる
//   cout << a << endl;

//   string s, t;
//   getline(cin, s); // 変数sで入力を一行受け取る
//   getline(cin, t); // 変数tで入力を一行受け取る
 
//   cout << "一行目 " << s << endl;
//   cout << "二行目 " << t << endl;


  string S;
  cin >> S;

  // ここにプログラムを追記
  
  int i, result = 1;
  for(i = 0; i < S.size(); i++){
    if(S.at(i) == '+'){
        result += 1;
    }else if(S.at(i) == '-'){
        result -= 1;
    }
  }

  cout << result << endl;

}
