# ゼロから作る Deep Learning

---

![表紙](https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png)

---

本リポジトリはオライリー・ジャパン発行書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』を参考にしました。

## ファイル構成

|フォルダ名 |説明                         |
|:--        |:--                          |
|cmake      |外部依存関係に関する cmake    |
|src        |本体のソースコード    |
|tests      |テストのソースコード    |

## 必要条件

* c++14
* cmake
* make

## 実行方法

### build
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### test
(上記で build したディレクトリで)
```
$ make test
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。