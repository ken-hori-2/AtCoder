// #include <bits/stdc++.h>
// using namespace std;

#include <iostream>
  using std::cin;
  using std::cout;
  using std::endl;
  using std::string;

int main() {
  // 変数a,b,cにtrueまたはfalseを代入してAtCoderと出力されるようにする。
  bool a = true; // true または false
  bool b = false; // true または false
  bool c = true; // true または false

  // ここから先は変更しないこと

  if (a) { // a = True
    cout << "At";
  }
  else {
    cout << "Yo";
  }

  if (!a && b) { // a = False, b  = True
    cout << "Bo";
  }
  else if (!b || c) { // b = False, c = True
    cout << "Co";
  }

  if (a && b && c) {
    cout << "foo!";
  }
  else if (true && false) {
    cout << "yeah!";
  }
  else if (!a || c) { // a = False, c = True
    cout << "der";
  }

  cout << endl;
}
