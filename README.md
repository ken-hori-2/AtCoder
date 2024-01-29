# AtCoder（競技プログラミング）
競技プログラミングの記録用リポジトリです.

# 基本的な手順
    # リモートリポジトリを先に作った場合
        <github上でリポジトリ作成 & url取得>
        * git clone https://github.com/ken-hori-2/AtCoder.git
        * git init & git remote add ~ も含まれる気がする
    # ローカルリポジトリを先に作った場合
        git init
        ~
        git remote add origin git@github.com:ken-hori-2/AtCoder.git # リモートリポジトリをoriginに設定
    
    # 以降は以下のコマンドを参考に実行

<!-- # Features

"hoge"のセールスポイントや差別化などを説明する -->

# Requirement

* git version 2.39.3 (Apple Git-145)
* Python 3.8.8

# Installation

Requirementで列挙したライブラリなどのインストール方法を説明する

```bash
pip install huga_package
```

<!-- # Usage

```bash
git clone https://github.com/ken-hori-2/AtCoder.git
cd AtCoder
``` -->

<!-- # Note -->

# 確認コマンド
    <基本コマンド>
    git status # gitで管理している/していないファイルを確認
        > 緑色の状態 : ステージング(ファイルがgitの管理下にある状態)されている
    git add . # 全てのファイルをステージング (指定ファイルをgitで管理する)
    git commit -m "コメント" # gitに保存 (変更点の保存/セーブポイント)
    
    git log # 何時何分に誰がコミットしたかわかる
    git push origin main # 保存したコミットをアップロード
    
    <最新情報をダウンロード>
    git pull origin # リモートリポジトリの最新情報をローカルに保存

    <ブランチの作成>
    git branch {ブランチ名} # ブランチの作成
    git checkout {ブランチ名} # 現在のブランチを切り替える
    git branch # ブランチ一覧を表示
        > 緑色が現在のブランチ
    
    <マージ>
    git merge {マージしたいブランチ名} # マージ先のブランチに移動する
        > mainにdevelopをマージさせたい場合 : mainに移動して "git merge develop"
        > mergeはコマンドでやる方法
    github上でマージする場合 : ブランチをアップロードしてpullリクエストをする
        > わざわざローカルに最新情報をpullしなくてよくなる & メモも残せる
        > こっちが主流になりつつあるそう
    
    <マージ前にローカルにリモートの最新情報を反映>
    git fetch origin # 正統な歴史ではなくパラレルワールド状態になってしまったときに行う
        > リモートの最新情報を取得しローカルの自分の開発しているものにマージする
    git branch -a # リモート側も含めた全てのブランチを表示
    git merge origin/{ブランチ名} # リモートのブランチをローカルのブランチにマージ
        > viエディタが起動するのでメモを残す
        > パラレルワールドが正統な歴史と合流
    > git pull {ブランチ名} # 上の二つのコマンドのショートカット
        注意事項:
            pullするリモートの内容が自分の変更箇所と被っている場合:
                git側はどっちを優先すればいいかわからないのでコンフリクトする
            
            コンフリクトしているファイルの中身ギザギアマークが追加されている
            1. ギザギザ文字を消す
            2. 同じ箇所(行)の変更を修正する
            3. git commit -m "コメント"

    git push origin # リモートにアップロード

    <上記の機能に加えて歴史を改ざん>
    git fetch origin
    git rebase origin/{ブランチ名} {ブランチ名} # origin/{ブランチ名}を {ブランチ名}にリベースする
    git push origin -f # 強制的にアップロード(ローカルが正)
        >綺麗な一直線に歴史を改ざんできる
    
    
    <変更内容を最終コミットの時点まで戻す>
    # 単純に変更取り消し
    1. git reset --hard HEAD

    # 本来開発するはずのブランチと異なるブランチで開発してしまっていた場合
    1. git stash # resetと同じ様に変更部分を取り消し + メモリ上に保存 
    2. git checkout {ブランチ名}
    3. git stash pop # メモリ上に保存していた変更点を現在のブランチ上に追加
    
# 補足
    # ローカルファイルが自分の変更箇所以外(pullしたベース部分)も違うことによるエラー...pushするときにリモートが他の誰かが変更して自分のローカルのものと歴史が変わっていた場合(最初のpull時と違う)
    # 初級者向け(fetch + maerge = pull)
        # リモートの内容を手元にマージ
        1. git fetch origin　<pushする先のbranch(mainや開発branch[今回はdevelop])の最新情報をリモートからダウンロード>
        2. git merge origin/branch名(mainとかdevelop) <origin/develop = リモート上のdevelopを手元のdevelopにマージ (=ローカルに反映)>
            コメントを残す必要があるので viエディタが起動する. -> :wq で保存
    
        # 上記の二つのコマンドを一つでできるショートカットコマンド
        1. git pull develop (developはマージしたいブランチ名)
    
    #上級者向け(fetch + rebase)... コミットツリーを一列にできる = コミットの歴史を改ざんできる
        # リモートの内容を手元にマージ
        1. git fetch origin
        2. git rebase origin/develop develop
        3. git push origin -f <強制的にローカルの内容をリモートに上書き>
    
    ※ 自分の変更箇所以外の歴史(pull時のベース部分)リモートの最新状態に更新してくれる
    ※ ただし、自分の変更点と同じ箇所が変わっている場合は、以下のコンフリクトの対処法を行う

    # コンフリクトする...共同開発で同じファイルの同じ行を編集してpushしようとした場合

    # 要はリモート側にpushされた "同じ部分の異なる変更" のどっちが正しいのか見分けがついていない状態


# 変更内容を最終コミットの時点まで戻す

    # 単純に変更取り消し
    1. git reset --hard HEAD

    # 本来開発するはずのブランチと異なるブランチで開発してしまっていた場合
    1. git stash # resetと同じ様に変更部分を取り消し + メモリ上に保存 
    2. git checkout branch名
    3. git stash pop # メモリ上に保存していた変更点を現在のブランチ上に追加


# まとめ
一番最初はclone, それ以降はpullでリモートの最新状態をローカルにダウンロード

# Author

* 作成者 ken
* 所属
* E-mail stmuymte@gmail.com

# License

<!-- "hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

"hoge" is Confidential. -->

# 参考資料
```bash
AtCoder_memo.txt
```