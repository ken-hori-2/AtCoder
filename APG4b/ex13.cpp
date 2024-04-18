#include <bits/stdc++.h>
using namespace std;

// 例題
// int main(){
//     int N;
//     cin >> N;
    
//     std::vector<int> math_points(N);
//     std::vector<int> eng_points(N);

    
//     for(int i=0; i<N; i++){
//         cin >> math_points.at(i);
//     }
//     for(int i=0; i<N; i++){
//         cin >> eng_points.at(i);
//     }

//     for(int i=0; i<N; i++){
//         std::cout << math_points.at(i) + eng_points.at(i) << std::endl;
//     }

//     // 初期化はどちらの書き方でもいい
//     // 上書きも可能
//     std::vector<int> vec(3, 10); // {10, 10, 10} で初期化
//     vec = std::vector<int>(100, 2); // 100要素の配列 {2, 2, ... , 2, 2} で上書き
//     cout << vec.at(99) << endl;

//     return 0;
// }

// 問題
int main() {
  int N;
  cin >> N;

  // N要素の配列
  std::vector<int> test_points(N);

  // 入力処理
  for(int i = 0; i < N; i++){
    std::cin >> test_points.at(i);
  }

  // 合計点
  int sum=0; //   int average;
  // 合計点を計算
  for(int i = 0; i < N; i++){
    sum += test_points.at(i);
  }
  // 平均点
  int average = sum/N;

  // 平均点から何点離れているかを計算して出力 ... if分で条件分岐するやり方もあり
  for(int i = 0; i < N; i++){
    std::cout << std::abs(test_points.at(i) - average) << std::endl;
  }
}
