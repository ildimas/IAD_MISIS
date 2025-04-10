class PriceCalculation:
    def __init__(self, appartment_list, analog_appartment_list):
        self.analog_appartment_list = analog_appartment_list
        self.appartment_list = appartment_list
        self.floors_number_diff = 0.01           # процент на разницу в этажах
        self.item_estate_type_diff = 0.05        # процент на тип дома

        self.floor_location_diff = [             # процент по различию этажей
        #   О/А ПЕРВЫЙ эт.  О/А СРЕДНИЙ эт.    О/А ПОСЛЕДНИЙ эт.
            [0,                      -0.07,             -0.031],   # О/О ПЕРВЫЙ эт.      
            [0.075,                      0,              0.042],   # О/О СРЕДНИЙ эт.     
            [0.032,                  -0.04,                  0],   # О/О ПОСЛЕДНИЙ эт.     
        ]

        self.appartment_square_diff = [          # процент по различию площадей квартир     
        #   О/А <30 кв. м.  О/А 30-50 кв. м.  О/А 50-65 кв. м. О/А 65-90 кв. м. О/А 90-120 кв. м.  О/А >120 кв. м.
            [0,                 0.06,              0.14,            0.21,               0.28,           0.31],      # О/О <30 кв. м.
            [-0.06,                0,              0.07,            0.14,               0.21,           0.24],      # О/О 30-50 кв. м.
            [-0.12,            -0.07,                 0,            0.06,               0.13,           0.16],      # О/О 50-65 кв. м.
            [-0.17,            -0.12,             -0.06,               0,               0.06,           0.09],      # О/О 65-90 кв. м.
            [-0.22,            -0.17,             -0.11,           -0.06,                  0,           0.03],      # О/О 90-120 кв. м.
            [-0.24,            -0.19,             -0.13,           -0.08,              -0.03,              0],      # О/О >120 кв. м.
        ]

        self.appartment_square_diff_check = [30, 50, 65, 90, 120]

        self.kitchen_square_diff = [             # процент по различию площадей кухонь
        #   О/А до 7 кв. м.     О/А 7-10 кв. м.     О/А 10-15 кв. м.
            [   0,              -0.029,             -0.083],    # О/О до 7 кв. м.
            [0.03,                   0,             -0.055],    # О/О 7-10 кв. м.
            [0.09,               0.058,                  0],    # О/О 10-15 кв. м.
        ]

        self.kitchen_square_diff_check = [7, 10, 15]

        self.appartment_balcony_lodge_diff = [   # процент по наличию балкона / лоджии
        # О/А нет балкона / лоджии   О/А есть балкона / лоджии
            [0,                             -0.05],     # О/О нет балкона / лоджии
            [0.053,                             0],     # О/О есть балкон / лоджия
        ]

        self.item_minutes_for_subway_diff = [    # процент по кол-во минут от метро
        #   О/А до 5 мин.    О/А 5-10 мин.  О/А 10-15 мин.  О/А 15-30 мин.  О/А 30-60 мин.  О/А 60-90 мин.
            [    0,          0.07,          0.12,           0.17,           0.24,           0.29],    # О/О до 5 мин.
            [-0.07,             0,          0.04,           0.09,           0.15,            0.2],    # О/О 5-10 мин.
            [-0.11,         -0.04,             0,           0.05,           0.11,           0.15],    # О/О 10-15 мин.
            [-0.15,         -0.08,         -0.05,              0,           0.06,            0.1],    # О/О 15-30 мин.
            [-0.19,         -0.13,          -0.1,          -0.06,              0,           0.04],    # О/О 30-60 мин.
            [-0.22,         -0.17,         -0.13,          -0.09,          -0.04,              0],    # О/О 60-90 мин.
        ]

        self.item_minutes_for_subway_diff_check = [5, 10, 15, 30, 60, 90]

        self.appartment_renovation_type_diff = [
        #   О/А без отделки    О/А эконом    О/А улучшенный
            [0,                 -13400,         -20100],    # О/О без отделки
            [13400,                  0,          -6700],    # О/О эконом
            [20100,               6700,              0],    # О/О улучшенный
        ]

        self.appartment_walls_type_diff = [
        # панельный     монолитный    кирпичный
            [0,             -0.13,      -0.21],      # панельный
            [0.15,              0,     -0.092],      # монолитный
            [0.26,         -0.096,          0],      # кирпичный
        ]
    
        self.floors_difference()
        self.item_estate_type_difference()
        self.item_minutes_for_subway_difference()
        self.appartment_square_difference()
        self.kitchen_square_difference()
        self.floor_location_difference()
        self.appartment_balcony_lodge_difference()
        self.appartment_renovation_type_difference()
        self.appartment_walls_type_difference()

    def floors_difference(self):
        if self.analog_appartment_list[7] > self.appartment_list[6]:
            self.analog_appartment_list[1] *= (1 - self.floors_number_diff) ** (self.analog_appartment_list[7] - self.appartment_list[6])
        else:
            self.analog_appartment_list[1] *= (1 + self.floors_number_diff) ** (self.appartment_list[6] - self.analog_appartment_list[7])

    def item_estate_type_difference(self):
        if self.analog_appartment_list[0] == 'новостройка' and self.appartment_list[0] == 'вторичная':
            self.analog_appartment_list[1] *= (1 - self.item_estate_type_diff)
        elif self.analog_appartment_list[0] == 'вторичная' and self.appartment_list[0] == 'новостройка':
            self.analog_appartment_list[1] *= (1 + self.item_estate_type_diff)

    def item_minutes_for_subway_difference(self):
        for i in range(len(self.item_minutes_for_subway_diff_check)):
            if self.analog_appartment_list[2] <= self.item_minutes_for_subway_diff_check[i]:
                item_minutes_for_subway_diff_analog = i
                break
        
        for i in range(len(self.item_minutes_for_subway_diff_check)):
            if self.appartment_list[1] <= self.item_minutes_for_subway_diff_check[i]:
                item_minutes_for_subway_diff_appartment = i
                break
        
        self.analog_appartment_list[1] *= (1 + self.item_minutes_for_subway_diff[item_minutes_for_subway_diff_appartment][item_minutes_for_subway_diff_analog])

    def appartment_square_difference(self):
        for i in range(len(self.appartment_square_diff_check)):
            if self.analog_appartment_list[4] < self.appartment_square_diff_check[i]:
                appartment_square_diff_analog = i
                break
        if self.analog_appartment_list[4] >= 120:
            appartment_square_diff_analog = 5

        for i in range(len(self.appartment_square_diff_check)):
            if self.appartment_list[3] < self.appartment_square_diff_check[i]:
                appartment_square_diff_appartment = i
                break
        if self.appartment_list[3] >= 120:
            appartment_square_diff_appartment = 5
        
        self.analog_appartment_list[1] *= (1 - self.appartment_square_diff[appartment_square_diff_appartment][appartment_square_diff_analog])

    def kitchen_square_difference(self):
        try:
            for i in range(len(self.kitchen_square_diff_check)):
                if self.analog_appartment_list[5] <= self.kitchen_square_diff_check[i]:
                    kitchen_square_diff_analog = i
                    break
            
            for i in range(len(self.kitchen_square_diff_check)):
                if self.appartment_list[4] <= self.kitchen_square_diff_check[i]:
                    kitchen_square_diff_appartment = i
                    break
            
            self.analog_appartment_list[1] *= (1 + self.kitchen_square_diff[kitchen_square_diff_appartment][kitchen_square_diff_analog])
        except UnboundLocalError:
            self.analog_appartment_list[1] *= 1 

    def floor_location_difference(self):
        if self.analog_appartment_list[6] == 1:
            floor_location_difference_analog = 0
        elif self.analog_appartment_list[6] == self.analog_appartment_list[7]:
            floor_location_difference_analog = 2
        else:
            floor_location_difference_analog = 1

        if self.appartment_list[5] == 1:
            floor_location_difference_appartment = 0
        elif self.appartment_list[5] == self.appartment_list[6]:
            floor_location_difference_appartment = 2
        else:
            floor_location_difference_appartment = 1

        self.analog_appartment_list[1] *= (1 + self.floor_location_diff[floor_location_difference_appartment][floor_location_difference_analog])

    def appartment_balcony_lodge_difference(self):
        if self.analog_appartment_list[8] == False and self.appartment_list[7] == True:
            self.analog_appartment_list[1] *= (1 + self.appartment_balcony_lodge_diff[1][0])
        elif self.analog_appartment_list[8] == True and self.appartment_list[7] == False:
            self.analog_appartment_list[1] *= (1 + self.appartment_balcony_lodge_diff[0][1])

    def appartment_renovation_type_difference(self):
        if self.analog_appartment_list[9] == 'без отделки':
            appartment_renovation_type_difference_analog = 0
        elif self.analog_appartment_list[9] == 'эконом':
            appartment_renovation_type_difference_analog = 1
        elif self.analog_appartment_list[9] == 'улучшенный':
            appartment_renovation_type_difference_analog = 2

        if self.appartment_list[8] == 'без отделки':
            appartment_renovation_type_difference_appartment = 0
        elif self.appartment_list[8] == 'эконом':
            appartment_renovation_type_difference_appartment = 1
        elif self.appartment_list[8] == 'улучшенный':
            appartment_renovation_type_difference_appartment = 2

        self.analog_appartment_list[1] += self.appartment_renovation_type_diff[appartment_renovation_type_difference_appartment][appartment_renovation_type_difference_analog]

    def appartment_walls_type_difference(self):
        try:
            if self.analog_appartment_list[10] == 'панельный':
                appartment_walls_type_difference_analog = 0
            if self.analog_appartment_list[10] == 'монолитный':
                appartment_walls_type_difference_analog = 1
            if self.analog_appartment_list[10] == 'кирпичный':
                appartment_walls_type_difference_analog = 2

            if self.appartment_list[9] == 'панельный':
                appartment_walls_type_difference_appartment = 0
            elif self.appartment_list[9] == 'монолитный':
                appartment_walls_type_difference_appartment = 1
            elif self.appartment_list[9] == 'кирпичный':
                appartment_walls_type_difference_appartment = 2

            self.analog_appartment_list[1] *= (1 + self.appartment_walls_type_diff[appartment_walls_type_difference_appartment][appartment_walls_type_difference_analog])
        except UnboundLocalError:
            self.analog_appartment_list[1] *= 1


if __name__ == "__main__":
    dict = {'1': ['новостройка', 15000000, 5, 2, 64, 8, 13, 15, False, 'эконом', 'монолитный', 0.8], 
            '2': ['вторичная', 10000000, 3, 2, 70, 8, 14, 14, True, 'эконом', 'панельный', 1.2],
            }
    #квартира над которой проходит расчет
    appartment_info = ['вторичная', 10, 2, 71, 7, 12, 19, True, 'улучшенный', 'кирпичный']
    # appartment_info[1] - расстояние до метро
    # appartment_info[2] - количество комнат
    # appartment_info[3] - площадь квартиры
    # appartment_info[4] - площадь кухни
    # appartment_info[5] - этаж квартиры
    # appartment_info[6] - этажность здания

    price_less_than_km = 0              # сумма цен квартир в пределах 1 км
    price_less_than_km_num = 0          # кол-во квартир в пределах 1 км
    price_more_than_km = 0              # сумма цен квартир в пределах 3 км
    price_more_than_km_num = 0          # кол-во квартир в пределах 3 км
    final_appartment_price = 0          # ФИНАЛЬНАЯ цена за квартиру   

    for key, value in dict.items():
        a = PriceCalculation(appartment_info, dict[key])
        if value[11] <= 1:
            price_less_than_km += value[1]
            price_less_than_km_num += 1
        else:
            price_more_than_km += value[1]
            price_more_than_km_num += 1

    if price_less_than_km_num == 0:
        final_appartment_price += price_more_than_km / price_more_than_km_num
    elif price_more_than_km_num == 0:
        final_appartment_price += price_less_than_km / price_less_than_km_num
    else:
        final_appartment_price += ((price_less_than_km / price_less_than_km_num) * 1.2 + (price_more_than_km / price_more_than_km_num) * 0.8) / 2
    
    final_appartment_price = round(final_appartment_price * 0.955)
    print(final_appartment_price)