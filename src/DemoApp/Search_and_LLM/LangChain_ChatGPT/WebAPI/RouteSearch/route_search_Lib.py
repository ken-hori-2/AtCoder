# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import strip
import datetime

# #出発駅の入力
# departure_station = input("出発駅を入力してください：")
# #到着駅の入力
# destination_station = input("到着駅を入力してください：")

from langchain.tools.base import BaseTool

class RouteSearch(): # BaseTool): # BaseToolの記述がなくても動く
    
    # def run(self, departure_station, destination_station,     shinkansen, serach_results_priority): # オプションの引数ありバージョン
    
    # dt_now_arg を引数に指定しない場合はコメントアウトにする
    # def run(self, dt_now_arg, departure_station, destination_station,     shinkansen, serach_results_priority): # オプションの引数ありバージョン
    def run(self, dt_now, departure_station, destination_station,     shinkansen, serach_results_priority): # オプションの引数ありバージョン
        #経路の取得先URL
        # route_url = "https://transit.yahoo.co.jp/search/print?from="+departure_station+"&flatlon=&to="+ destination_station

        """
        # 以前の現在時刻で検索する方法（日時指定なし）
        # オプションの引数ありバージョン
        route_url = "https://transit.yahoo.co.jp/search/print?from="+departure_station+"&flatlon=&to="+destination_station+"&shin="+shinkansen +"&s="+serach_results_priority # 0 or 1 or 2
        """

        
        
        
        #############################
        # 2024/6/17 日時指定version #
        #############################
        # dt_now = datetime.datetime(2024, 6, 17, 8, 00) # 30)
        # # dt_now = datetime.datetime(2024, 6, 7, 18, 00)
        # dt_now = dt_now_arg

        # dt_now_arg を引数に指定しない場合はコメントアウトにする
        # dt_now = datetime.datetime.strptime(dt_now_arg, "%Y-%m-%d %H:%M:%S") # 文字列からdatetime型
        # start_date = datetime.datetime(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute)


        start_date = datetime.datetime(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute)
        print("Boarding Date and Time:", start_date)
        # yearは4桁になるので大丈夫
        yyyy = f'{dt_now.year}'
        """"""
        # 0を追加してエラー回避 # 1桁では挙動がおかしくなる（時刻がずれる）
        if len(f'{dt_now.month}') > 1:
            mm = f'{dt_now.month}'
        else:
            mm = f'0{dt_now.month}'
        
        if len(f'{dt_now.day}') > 1:
            dd = f'{dt_now.day}'
        else:
            dd = f'0{dt_now.day}'

        if len(f'{dt_now.hour}') > 1:
            hh = f'{dt_now.hour}'
        else:
            hh = f'0{dt_now.hour}' # 0を追加してエラー回避（18時は018時でもOK） # hh = f'{dt_now.hour}' # 1桁ではエラーになる
        """"""
        # minuteは2桁になるので大丈夫
        m1 = f'{int(dt_now.minute/10)}'
        m2 = f'{int(dt_now.minute%10)}'
        # # print("test:", dt_now.minute, dt_now.minute/10, dt_now.minute%10)
        # print(yyyy, mm, dd, hh, m1, m2)
        # # print(len(yyyy), len(mm), len(dd), len(hh), len(m1), len(m2))
        # https://transit.yahoo.co.jp/?from=玉川学園前&flatlon=&to=大崎&via=&viacode=&y=2024&m=06&d=17&hh=08&m1=3&m2=1&type=1&s=0&ws=3&expkind=1&ticket=ic&no=1&fromgid=&togid=&tlatlon=&userpass=1&al=1&shin=1&ex=1&hb=1&lb=1&sr=1
        route_url = "https://transit.yahoo.co.jp/search/print?from="+departure_station+"&flatlon=&to="+destination_station+"&via=&viacode=&y="+yyyy+"&m="+mm+"&d="+dd+"&hh="+hh+"&m1="+m1+"&m2="+m2 + "&shin="+shinkansen +"&s="+serach_results_priority
        #############################
        # 2024/6/17 日時指定version #
        #############################


        


        # print(route_url)
        #Requestsを利用してWebページを取得する
        route_response = requests.get(route_url)

        # BeautifulSoupを利用してWebページを解析する
        route_soup = BeautifulSoup(route_response.text, 'html.parser')

        #経路のサマリーを取得
        route_summary = route_soup.find("div",class_ = "routeSummary")
        #所要時間を取得
        required_time = route_summary.find("li",class_ = "time").get_text()
        #乗り換え回数を取得
        transfer_count = route_summary.find("li", class_ = "transfer").get_text()
        #料金を取得
        fare = route_summary.find("li", class_ = "fare").get_text()

        # print("======"+departure_station+"から"+destination_station+"=======")
        # print("所要時間："+required_time)
        # print(transfer_count)
        # print("料金："+fare)

        #乗り換えの詳細情報を取得
        route_detail = route_soup.find("div",class_ = "routeDetail")

        #乗換駅の取得
        stations = []
        stations_tmp = route_detail.find_all("div", class_="station")
        for station in stations_tmp:
            stations.append(station.get_text().strip())
            # print(station.get_text().strip())

        #乗り換え路線の取得
        lines = []
        lines_tmp = route_detail.find_all("li", class_="transport")
        for line in lines_tmp:
            line = line.find("div").get_text().strip()
            lines.append(line)

        #路線ごとの所要時間を取得
        estimated_times = []
        estimated_times_tmp = route_detail.find_all("li", class_="estimatedTime")
        for estimated_time in estimated_times_tmp:
            estimated_times.append(estimated_time.get_text())

        # print(estimated_times)

        #路線ごとの料金を取得
        fars = []
        fars_tmp = route_detail.find_all("p", class_="fare")
        for fare in fars_tmp:
            fars.append(fare.get_text().strip())


        #乗り換え詳細情報の出力
        print("======乗り換え情報======")
        # for station,line,estimated_time,fare in zip(stations,lines,estimated_times,fars):
        #     print(station)
        #     print( " | " + line + " " + estimated_time + " " + fare)

        # このやり方だと一番少ない個数のリストの回数に合わせられてしまうのでダメ
        # for station,line,fare in zip(stations,lines,fars):
        #     print(station)
        #     print( " | " + line + " " + fare)

        # print(stations[len(stations)-1])


        # for station in stations:
        #     print(station)
        #     print( " | " + line + " " + estimated_time + " " + fare)
        # print("**********")
        # print(stations)
        # print(lines)
        # print(estimated_times)
        # print(fars)
        # print("**********")
        # for i in range(len(stations)):
        #     print(stations[i])
        #     try:
        #         print( " | " + lines[i] + " " + fars[i])
        #     except:
        #         try:
        #             print( " | " + lines[i] + " ---")
        #         except:
        #             pass
        
        ret = []
        for i in range(len(stations)):
            ret.append(stations[i])
            # try:
            #     # ret.append( " | " + lines[i] + " " + fars[i])
            #     ret.append(" " +lines[i] + " " + fars[i])
            # except:
            #     try:
            #         # ret.append( " | " + lines[i] + " ---")
            #         ret.append(" " + lines[i])
            #     except:
            #         pass
            try:
                ret.append(lines[i] + " " + fars[i])
            except:
                try:
                    ret.append(lines[i])
                except:
                    pass
        
        return "\n\n".join(ret)[: 300]

if __name__ == "__main__":

    # test
    # https://transit.yahoo.co.jp/search/result?from=玉川学園前&to=大崎&fromgid=
    # &togid=&flatlon=%2C%2C22804&tlatlon=%2C%2C22559
    # &via=&viacode=&y=2024&m=06&d=17&hh=08&m1=2&m2=9
    # &type=1&ticket=ic&expkind=1
    # &userpass=1&ws=3
    # &s=0&al=1&shin=1&ex=1&hb=1&lb=1&sr=1

    #出発駅の入力
    departure_station = input("出発駅を入力してください：")
    #到着駅の入力
    destination_station = input("到着駅を入力してください：")

    yahoo_serach = RouteSearch() # departure_station, destination_station)
    # result = yahoo_serach.run(departure_station, destination_station)
    dt_now = str(datetime.datetime(2024, 6, 17, 8, 00))
    # dt_now = datetime.datetime(2024, 6, 17, 8, 00)
    result = yahoo_serach.run(dt_now, departure_station, destination_station,     "0", "0")
    print(result)
