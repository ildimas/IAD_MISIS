import requests
from bs4 import BeautifulSoup
import geocoder

class Application:
    def __init__(self):
        self.starter = 1
        self.end = 5
        self.all_appartments_dict = {}
        self.pages = 86
        self.url_s = 'https://www.russianrealty.ru/%D0%9F%D1%80%D0%BE%D0%B4%D0%B0%D0%B6%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80/'
    

    # метод для запроса страницы и её парс через bs4
    def request_reader(url):
        headers = {
            "Accept": 
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0",
        }

        req = requests.get(url, headers=headers)
        src = req.text

        with open("index.html", "w", encoding="utf-8") as file:
            file.write(src)

        with open("index.html", "r", encoding="utf-8") as file:
            src = file.read()

        file.close()
        soup = BeautifulSoup(src, "lxml")        
        return soup

        
    def parser(self):
        if self.end >= self.pages:
            return self.all_appartments_dict
        else:
            for i in range(self.starter, self.end):
                # Ссылки на каталоги типов недвижимости, вторичная (1) и новостройки (2):
                # 1) https://www.russianrealty.ru/%D0%9F%D1%80%D0%BE%D0%B4%D0%B0%D0%B6%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80/{i}/
                # 2) https://www.russianrealty.ru/%D0%9F%D1%80%D0%BE%D0%B4%D0%B0%D0%B6%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80-%D0%B2-%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B9%D0%BA%D0%B5/{i}/

                url = self.url_s + str(i) + '/'
                
                # тип недвижимости - зависит от ссылки выше
                if url.find("%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B9%D0%BA%D0%B5") != -1:
                    item_estate_type = 'новостройка'
                else:
                    item_estate_type = 'вторичная'

                    
                # чтение
                page = Application.request_reader(url)
                all_appartments_hrefs = page.find_all("div", {"class": "hproduct"})


                for div in all_appartments_hrefs:
                    # имя, ссылка, количество комнат
                    item_name = div.find('a').text
                    item_name = item_name.replace("&sup2", "2")
                    item_href = "https:" + div.find('a').get('href')
                    item_rooms_number = int(item_name[0])

                    # цена и адрес
                    item_price = int(div.find('span').text)
                    item_adress_details = div.find('p', class_ = "adr").text

                    try:
                        in_point_item_adress_details = item_adress_details.index("«")
                        item_adress = item_adress_details[0 : in_point_item_adress_details]
                    except ValueError:
                        item_adress = item_adress_details

                    # количество минут до метро
                    try:
                        in_point_item_adress_details_ending = item_adress_details[::-1].index(",")
                        item_minutes_for_subway = item_adress_details[len(item_adress_details) - in_point_item_adress_details_ending + 1 : len(item_adress_details) - 1]
                        in_point_item_minutes_for_subway = item_minutes_for_subway.index(" ")
                        item_minutes_for_subway = int(item_minutes_for_subway[: in_point_item_minutes_for_subway])
                    except ValueError:
                        item_minutes_for_subway = 15

                    # площадь квартиры и кухни
                    try:
                        in_point_item_name_beginning = item_name.index(" ")
                        in_point_item_name_ending = item_name[::-1].index(" ")
                        squares = item_name[in_point_item_name_beginning + 1 : len(item_name) - (in_point_item_name_ending + 1)]

                        in_point_squares_beginning = squares.index("/")
                        appartment_square = float(squares[0 : in_point_squares_beginning])
                        in_point_squares_ending = squares[::-1].index("/")
                        appartment_kitchen_square = float(squares[len(squares) - in_point_squares_ending :])
                    except ValueError:
                        appartment_square = 50
                        appartment_kitchen_square = 10

                    appartment_info = div.find_all('li') # список li элементов, через которые берётся информация ниже
                    
                    #координаты
                    try:
                        y = geocoder.osm(item_adress)
                        item_latitude_coordinates = (y.latlng)[0]
                        item_longtitude_coordinates = (y.latlng)[1]
                    except TypeError:
                        item_longtitude_coordinates = 666
                        item_latitude_coordinates = 666
                    # этаж квартиры и дома
                    try:
                        appartment_floors_info = appartment_info[0].text

                        in_point_appartment_floors_info_sliced = appartment_floors_info.index("-")
                        appartment_floors_info = appartment_floors_info[: in_point_appartment_floors_info_sliced]

                        in_point_appartment_floors_info_beginning = appartment_floors_info.index(" ")
                        appartment_floor = int(appartment_floors_info[0 : in_point_appartment_floors_info_beginning])
                        in_point_appartment_floors_info_ending = appartment_floors_info[::-1].index(" ")
                        appartment_floors_in_building = int(appartment_floors_info[len(appartment_floors_info) - in_point_appartment_floors_info_ending :])
                    except ValueError:
                        appartment_floor = 5
                        appartment_floors_in_building = 10
                    # наличие балкона или лоджии
                    appartment_balcony_lodge = appartment_info[1].text
                    if appartment_balcony_lodge.find("балкон") != -1 or appartment_balcony_lodge.find("лоджия") != -1:
                        appartment_balcony_lodge = True
                    else:
                        appartment_balcony_lodge = False

                    # тип ремонта
                    try:
                        if appartment_balcony_lodge:
                            appartment_renovation_type = appartment_info[2].text
                        else:
                            appartment_renovation_type = appartment_info[1].text
                        in_point_appartment_renovation_type = appartment_renovation_type.index(" ")
                        appartment_renovation_type = appartment_renovation_type[in_point_appartment_renovation_type + 1 : ]

                        if appartment_renovation_type == "требуется ремонт" or appartment_renovation_type == "без отделки":
                            appartment_renovation_type = "нет ремонта"
                        elif appartment_renovation_type == "косметический" or appartment_renovation_type == "под чистовую":
                            appartment_renovation_type = "эконом"
                        else:
                            appartment_renovation_type = "улучшенный"
                    except ValueError:
                        appartment_renovation_type = 'нет ремонта'


                    # тип дома
                    try:
                        page = Application.request_reader(item_href)
                        appartment_card = page.find("div", {"class": "col-lg-4 col-md-6 col-sm-12 desc-list"})
                        appartment_details = appartment_card.find('ul').text

                        appartment_house_type = 'not stated'
                        if appartment_details.find("Тип дома:") != -1:
                            appartment_house_type = appartment_details[appartment_details.find("Тип дома:") :]
                            in_point_appartment_house_type_beginning = appartment_house_type.index(":")
                            in_point_appartment_house_type_ending = appartment_house_type.index("\n")
                            appartment_house_type = appartment_house_type[in_point_appartment_house_type_beginning + 2 : in_point_appartment_house_type_ending]
                    except ValueError:
                        appartment_house_type = 'not stated'

                    # словарь с ключом в виде названия квартиры и значением в виде списка переменных
                    self.all_appartments_dict[item_name] = [ item_href,                         # ссылка / string
                                                        item_estate_type,                       # тип недвижимости / string
                                                        item_price,                             # цена / int
                                                        item_adress,                            # адрес / string
                                                        item_minutes_for_subway,                # кол-во минут до метро / int ; при отсутствии - 'not stated'
                                                        item_rooms_number,                      # количество комнат / int
                                                        appartment_square,                      # общая площадь квартиры / float ; при отсутствии - 'not stated'
                                                        appartment_kitchen_square,              # площадь кухни / float ; при отсутствии - 'not stated'
                                                        appartment_floor,                       #! этаж квартиры / int
                                                        appartment_floors_in_building,          #! этажность дома / int
                                                        appartment_balcony_lodge,               # наличие балкона или лоджии / bool
                                                        appartment_renovation_type,             # тип ремонта / string
                                                        appartment_house_type,                  # тип дома / string
                                                        item_latitude_coordinates,              # Долгота / int
                                                        item_longtitude_coordinates             # Широта  / int
                                                        ]
            
            if self.end >= self.pages:
                print(f"Процент выполениния: 100%. Всего спаршено: {len(self.all_appartments_dict)}")
            else:
                print(f"Процент выполениния: {round((self.end / self.pages) * 100)}%. Всего спаршено: {len(self.all_appartments_dict)}")
            self.end += 5; self.starter += 5

            return self.parser()
        
        
if __name__ == "__main__":
    x = Application()
    pars_dict = x.parser()
    for key in pars_dict:
        main_list = pars_dict[key]
        print(main_list)
    y = Application()
    y.url_s = 'https://www.russianrealty.ru/%D0%9F%D1%80%D0%BE%D0%B4%D0%B0%D0%B6%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80-%D0%B2-%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B9%D0%BA%D0%B5/'
    new_dict = y.parser()
    for key in new_dict:
        ma_li = new_dict[key]
        print(ma_li)