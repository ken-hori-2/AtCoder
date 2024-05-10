# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
# import strip

# # #出発駅の入力
# # departure_station = input("出発駅を入力してください：")
# # #到着駅の入力
# # destination_station = input("到着駅を入力してください：")

# from langchain.tools.base import BaseTool

class RouteSearch(): # BaseTool): # BaseToolの記述がなくても動く
    
    def run(self, departure_station, destination_station,     shinkansen, serach_results_priority): # オプションの引数ありバージョン
        #経路の取得先URL
        # route_url = "https://transit.yahoo.co.jp/search/print?from="+departure_station+"&flatlon=&to="+ destination_station

        # オプションの引数ありバージョン
        route_url = "https://transit.yahoo.co.jp/search/print?from="+departure_station+"&flatlon=&to="+destination_station+"&shin="+shinkansen +"&s="+serach_results_priority # 0 or 1 or 2
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
            try:
                ret.append( " | " + lines[i] + " " + fars[i])
            except:
                try:
                    ret.append( " | " + lines[i] + " ---")
                except:
                    pass
        
        return "\n\n".join(ret)[: 300]

if __name__ == "__main__":
    #出発駅の入力
    departure_station = input("出発駅を入力してください：")
    #到着駅の入力
    destination_station = input("到着駅を入力してください：")

    yahoo_serach = RouteSearch() # departure_station, destination_station)
    result = yahoo_serach.run(departure_station, destination_station)
    print(result)
