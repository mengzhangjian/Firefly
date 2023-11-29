import json
import requests


def main():
    url = 'http://10.191.72.167:8802/firefly'
    timeout = 60    # 超时设置

    # 生成超参数
    max_new_tokens = 1024
    top_p = 0.85
    temperature = 0.35
    repetition_penalty = 1.0
    do_sample = True

    inputs = """
    沈阳必吃美食：
    锅包肉：具有浓厚东北特色的美食，独特的酸甜口感，让人上瘾。
    老边饺子：历史悠久的饺子，以其皮薄馅嫩、味道鲜美、汤汁丰富而著名。
    马家烧麦：沈阳著名的小吃店之一，以烧麦和羊肉为主要食材，口感鲜美。
    杨家吊炉饼：传统东北饼食，外酥里嫩，味道可口。
    李连贵熏肉大饼：沈阳的老字号小吃，熏肉大饼的口味独特，让人回味无穷。
    西塔大冷面：朝鲜族特色小吃，面条筋道，汤汁清爽，配菜丰富。
    沈阳回头：起源于清朝光绪年间的小吃，外皮金黄酥脆，内馅鲜美可口。
    老山记海城馅饼：特色馅饼，口感鲜美，是沈阳有名的早餐之一。
    康平羊汤：沈阳的羊汤十分有名，康平羊汤更是其中的佼佼者，汤汁浓郁，肉质鲜嫩。
    白肉血肠：东北传统美食，酸菜和血肠的搭配非常美味。
    三天两晚的旅游攻略：

    第一天：

    上午：到达沈阳站后，首先可以前往沈阳故宫游玩，感受历史的厚重。
    中午：在故宫附近的餐厅品尝当地的特色美食，如辽宁鱼头、炖酸菜等。
    下午：参观“九·一八”历史博物馆，了解中国的抗战历史。
    晚餐：在当地餐馆享用丰盛的晚餐，尝试沈阳的著名美食如锅包肉、老边饺子等。
    第二天：

    上午：参观张氏帅府，了解中国近现代历史。
    中午：在附近的餐厅品尝正宗的东北菜，如烧烤、炖菜等。
    下午：游览中街步行街，购物休闲两不误。
    晚餐：在中街附近的餐厅享用晚餐，尝试马家烧麦、杨家吊炉饼等沈阳特色小吃。
    第三天：

    上午：参观辽宁省博物馆，了解辽宁地区的历史文化。
    中午：在博物馆附近的餐厅享用午餐，尝试老山记海城馅饼等沈阳美食。
    下午：前往北陵公园游玩，感受沈阳的美丽自然风光。
    晚餐：在公园附近的餐厅享用晚餐，尝试康平羊汤、白肉血肠等特色美食。
    以上就是沈阳三天两晚的旅游攻略和必吃美食推荐，希望对您的旅行有所帮助！祝您旅途愉快！\n
    问: 请参照以上回复模板，告诉我成都必吃美食，以及制定一份成都三天两晚的旅游攻略"""  #和 请求内容
    
    inputs = inputs.strip()

    params = {
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    headers = {"Content-Type": "application/json", "Connection": "close"}
    response = requests.post(url, json=params, headers=headers)
    result = json.loads(response.text)['output']
    print(result)


if __name__ == '__main__':
    main()
