#include <bits/stdc++.h>
using namespace std;

int main() {
    int A, B, C;
    cin >> A >> B >> C;

    // int max_height = max(A, B, C); // 3つは比較できない ... 2つまで
    // int min_height = min(A, B, C); // 3つは比較できない ... 2つまで

    // std::cout << A << " " << B << " " << C << std::endl;
    
    std::vector<int> vec = {A, B, C};
    // std::vector<int> vec; // (3);
    // vec.push_back(A);
    // vec.push_back(B);
    // vec.push_back(C);

    // std::cout << vec.at(0) << " " << vec.at(1) << " " << vec.at(2) << std::endl;

    sort(vec.begin(), vec.end()); // 小さい順にソート 大きい順にしたいならreverseを使う
    // int max_height = vec.at(2); // 0);
    int min_height = vec.at(0); // 2);
    
    // vec.at(-1); // gccでは-1は使えないみたい
    // 代わりに以下のようにする
    int max_height = vec.at(vec.size() -1); // 今回はsize=3なのでvec.at(2);と同じ

    // std::cout << vec.at(0) << " " << vec.at(1) << " " << vec.at(2) << std::endl;

    // std::cout << std::abs(max_height - min_height) << std::endl;
    std::cout << max_height - min_height << std::endl;
}

// 解答例

// メモ
// min, max関数は2つしか引数を渡せないから使えないと思ってしまっていたが、以下のように順番に処理してあげることで使える

int main() {
    int a, b, c;
    cin >> a >> b >> c;

    int maximum = max(max(a, b), c); // 最大値を計算
    int minimum = min(min(a, b), c); // 最小値を計算

    cout << maximum - minimum << endl;
}
