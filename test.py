# coding=utf8
from transformers import BertTokenizer, AutoTokenizer
from utils import *
import torch
import os

def test_tokenizer():
	# tokenizer = MyTokenizer(BertTokenizer.from_pretrained('gpt-chinese-mini/vocab.txt'))
	tokenizer = AutoTokenizer.from_pretrained('yi-tokenizer', trust_remote_code=True)
	max_length = 500
	eos_token_id = 102
	pad_token_id = 0
	examples = {'instruction': ["法国的首都是什么？", '你是谁'], 'input': ['', ''],'output': ["法国的首都是巴黎。", 'I am a bot'], 'history': [[['我想去法国玩', '法国在欧洲'], ['真不错', '是呀']], []]}
	r = preprocess_sft_dataset_alpaca(examples, tokenizer, max_length, tokenizer.eos_token_id, tokenizer.pad_token_id)
	# print(r)
	idx = 1
	decoded = tokenizer._convert_id_to_token(r['input_ids'][idx])
	# print(len(decoded), len(r['input_ids'][0]), len(r['labels'][0]))
	# print(decoded)

	for input_id, label, d in zip(r['input_ids'][idx], r['labels'][idx], decoded):
		print(input_id, label, d if d != '\n' else '_n')

	# for x in r['input_ids']:
	# 	print('*' * 10)
	# 	print(x)
	# 	print(len(x))
	print(''.join([x for x in decoded if x not in ['<unk>']]))
	print(''.join([x for x in decoded if x not in ['<|startoftext|>', '<|endoftext|>', '<unk>']]))

def test_generate():
	config = read_json('gpt-chinese-mini/config.json')
	generation_config = read_json('gpt-chinese-large/generation_config.json')
	config['max_length'] = 10
	model = GPT(config, generation_config)
	input_ids = torch.ones(2, 5).long()
	response = model.generate(input_ids)
	print(response)

def test_stream():
	import time
	def generator():
		a = []
		for i in range(10):
			a.append(i)
			time.sleep(1)
			yield a

	def chat():
		for r in generator():
			yield r 

	start = 0
	for i in chat():
		print(i[start:], end='', flush=True)
		start = len(i)

def test_json(data_dir):
	from pyarrow import json as pjson
	for f in os.listdir(data_dir):
		print(f)
		if f.endswith('jsonl'):
			# data = read_jsonl(os.path.join(data_dir, f))
			pjson.read_json(os.path.join(data_dir, f))
			# del data

def test_dataset():
	from datasets import load_dataset
	from functools import partial
	# tokenizer = AutoTokenizer.from_pretrained('yi-tokenizer', trust_remote_code=True)
	for f in os.listdir('train/wikipedia'):
		print(f)
		dataset = load_dataset('json', data_files=[os.path.join('train/wikipedia', f)], split='train')
	# dataset = dataset.select(range(1000))
	# func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, max_length=1024)
	# dataset = dataset.map(func, batched=True, remove_columns=['text'], num_proc=10)
	# dataset = load_dataset('parquet', data_files=['train/en/train-00000-of-01650-f70471ee3deb09c0.parquet'], split='train')
	# dataset = dataset.select(range(1000))
	# func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, max_length=1024)
	# dataset = dataset.map(func, batched=True, remove_columns=['text'], num_proc=10)

	# dataset = load_dataset('parquet', data_files=['train/code/train-00000-of-00880.parquet'], split='train')
	# dataset = dataset.select(range(1000))
	# func = partial(preprocess_pretrain_code_dataset, tokenizer=tokenizer, max_length=1024)
	# dataset = dataset.map(func, batched=True, remove_columns=['code', 'repo_name', 'path', 'language', 'license', 'size'], num_proc=10)

	# print(dataset)
	# for row in dataset:
	# 	print(len(row['input_ids']))
	# 	print(row['input_ids'])
	# 	print(tokenizer.decode(row['input_ids']))
	# 	break
	# dataset = load_dataset('parquet', data_files=['train/en/train-00000-of-01650-f70471ee3deb09c0.parquet'], split='train', streaming=True)
	# func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, max_length=1024)
	# dataset = dataset.map(func, batched=True, remove_columns=['text'])
	# print(next(iter(dataset)))

def cal_model_size():
	from modeling_new import GPT
	c = 0
	config = read_json('configs/1B.json')
	model = GPT(config)
	print(model)
	for n, p in model.named_parameters():
		c += p.numel()
	print(c / 10 ** 9)

def test_concat():
	from baidubaike_preprocess import concat_all
	# row = {"title": "中国新音乐年鉴(2009)", "summary": None, "sections": [{"title": "基本信息", "content": "出版社: 上海音乐学院出版社; 第1版 (2011年4月1日) 外文书名: New Music in China 2009精装: 317页 正文语种: 简体中文 开本: 16 ISBN: 9787806924815 条形码: 9787806924815 商品尺寸: 26 x 19 x 2.6 cm 商品重量: 939 g 品牌: 上海音乐学院出版社 ASIN: B0051X92L4"}, {"title": "商品描述", "content": "编辑推荐钱仁平主编的《中国新音乐年鉴(2009)》收录了年度创作、重大项目、专题访谈和文献综述四个栏目，具体内容有：中央音乐学院现代音乐发展状况综述；第四届“帕拉天奴”杯作曲比赛综述；何训田访谈录：关于音乐大典《觉者之路》；作曲与作曲技术理论研究状况综述等。 \n目录一、年度创作\n中央音乐学院现代音乐发展状况综述 杜莹\n上海音乐学院音乐创作与作曲技术理论研究综述 王澍\n武汉音乐学院专业音乐创作综述 张瑕\n四川音乐学院新音乐创作、展演与研究情况综述 胡晓\n中国音乐学院专业音乐创作综述 孔祥怡\n沈阳音乐学院音乐创作综述 范哲明\n新音乐在西安 夏滟洲\n天津音乐学院音乐创作综述 顾之勉\n星海音乐学院音乐创作综述 房晓敏\n首都师范大学音乐学院音乐创作综述 蔡梦\n香港新音乐概况 陈锦标\n声动，不安于室——台湾音乐新创作 徐昭宇\n海外华人作曲家音乐创作、演出及出版述评 饶韵华\n二、重大项目\n现代音乐的锣鼓——写在“北京现代音乐节”六周年之际 项筱刚\n第二届当代音乐周综述 杨和平\n未来的挑战与音乐的回归——“2009上海音乐学院国际电子音乐周”后记 陈强斌 秦毅\n北京国际电子音乐节纪实 王鹤霏\n析音乐之结构 求学理之真谛——记“首届全国音乐分析学”学术盛会 贾达群 陈婷婷\n回顾交响，展望未来——“中国交响乐世纪回顾暨历届中国交响音乐季”纪事 李涛\n作曲大师与现代经典——德国当代音乐周回顾 夏苒\n第四届“帕拉天奴”杯作曲比赛综述 王颖\n第三届Con Tempo新室内乐作曲比赛综述 夏苒\n《中国当代作曲家曲库》系列活动述评 陈荃有\n三、专题访谈\n何训田访谈录：关于音乐大典《觉者之路》\n金湘访谈录：关于“金湘交响乐作品音乐会”\n王宁访谈录：关于歌剧《孔子》\n拉赫曼访谈录：关于“欧洲作曲家新作品音乐会”\n钱仁平访谈录：关于“华人作曲家手稿典藏与研究”项目与《中国新音乐年鉴》编撰工作\n四、文献综述\n作曲与作曲技术理论研究状况综述 钱仁平\n国际文献视野中的中国新音乐 韩斌 \n"}], "tags": [], "url": "http://baike.baidu.com/view/8180267.htm"}
	# row = {"title": "圣水濯足", "summary": "是由基督教传下的一种服侍礼节而来，耶稣为门徒洗脚，寓意为首的就是为仆，领导就是服侍的一种行为。", "sections": [], "tags": [], "url": "http://baike.baidu.com/view/8180245.htm"}
	row = {"title": "亡者之夜 U9周年贺岁版", "summary": "该地图是一款魔兽争霸3冰封王座的多人RPG防守类的地图，仅供10人玩，地图只支持魔兽争霸3冰封王座1.22版本", "sections": [{"title": "基本信息", "content": "【词条】：亡者之夜 U9周年贺岁版\n【拼音】：wang zhe zhi ye U9zhou nian gong he ban\n【性质】：魔兽防守地图 　　\n【地图版本】：U9周年恭贺版\n【适用游戏】：魔兽争霸3冰封王座v1.22版 　　\n【支持人数】： 1-10人"}, {"title": "使用说明", "content": "1.解压缩 \n2.把防守地图放至魔兽争霸安装目录Maps文件夹下，进入游戏后选择此地图即可。"}, {"title": "简介", "content": "《亡者之夜》U9周年贺岁版"}, {"title": "截图", "content": "  [1]\n  \n"}], "tags": ["魔兽", "魔兽争霸", "魔兽争霸3地图", "魔兽RPG"], "url": "http://baike.baidu.com/view/8180146.htm"}
	row = {"title": "美的FZ4010", "summary": None, "sections": [{"title": "重要参数", "content": "容积大小：4L 产品功率：900W 内胆材质：黄晶蜂窝内胆 预约定时煮饭：支持 其它性能：加热方式：三维立体加热 其它特点：1、2.0mm黄晶蜂窝玉铜釜 2、智... 产品尺寸：420*323*297mm 产品重量：6kg 电源性能：200V/50Hz 产品颜色：黑色"}, {"title": "主要参数", "content": "容积大小 4L\n产品功率 900W\n预约定时煮饭 支持\n内胆材质 黄晶蜂窝内胆\n产品重量 6kg\n产品尺寸 420*323*297mm\n电源性能 200V/50Hz\n产品颜色 黑色\n其它性能 加热方式：三维立体加热\n其它特点 1、2.0mm黄晶蜂窝玉铜釜\n2、智能触摸上盖操作\n3、不锈钢可拆洗盖板\n4、黑色喷涂弧形发热盘\n5、柔和开盖：白色大液晶显示屏;五步加热曲线，煮饭状态实时看\n6、耗电量显示：季节功能选择\n产品附件 \n包装清单电饭煲 x1\n量杯 x1\n饭铲 x1\n保修卡 x1\n说明书 x1\n"}], "tags": ["电饭煲"], "url": "http://baike.baidu.com/view/8628394.htm"}
	row = {'title': '123', "summary": None, 'sections': [{"title": "分集剧情", "content": "分集查询收起查询1集2集3集4集5集6集7集8集9集10集11集12集13集 第1集 　　1937年，河南安阳，中国政府派董济堂教授在全力发掘殷墟遗址。十年前考古专家光汉发现了龟背并讲起了它的故事，要看到天书必须将四片甲骨凑到一起。日本人知道消息后执行骷髅计划，他们要得到YM甲二十四坑埋藏的秘密，在日军未到达安阳前一切行动由樱花小组负责。　　日本电文中多次提出殷墟这个字被中国政府破解，组织上派古风以上海陈记古玩行的二掌柜身份去安阳收集古董。董济堂在YM甲二十四坑挖掘十几天发现了商代兵器，他知道文馆会有三件文物被盗窃，丢失和三件铜器都是商代中期的宝物，它们对研究商代礼祀有重要意义，警查挨家挨户地进行着搜查。"}, {"title": "全集在线观看", "content": ".vedio-content{background:#fafafb;padding:6px 0}.vedio-content ul{margin:0 3px;float:left;display:inline}.vedio-content ul li{float:left;margin:0 6px;display:inline}.vedio-content ul li a{line-height:2.5;text-align:center;font-size:12px;width:121px;display:block}.vedio-content ul li .pic-show{height:83px;position:relative;border:1px solid #ccc}.vedio-content ul li .pic-show:hover{border:1px solid #45a909}.vedio-content ul li img{height:77px;width:115px;margin:3px}.vedio-content a{text-decoration:none}.vedio-content a:hover{text-decoration:underline}.vedio-content .v-icon{background:transparent url(http://img.baidu.com/img/baike/play.png);position:absolute;left:10px;bottom:10px;height:18px;width:28px}* html .vedio-content .v-icon{background:0;filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(src='http://img.baidu.com/img/baike/play.png')}.vedio-content .new-icon{background:transparent url(http://img.baidu.com/img/baike/new.png) no-repeat;position:absolute;left:83px;top:0;height:39px;width:38px}* html .vedio-content .new-icon{background:0;filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(src='http://img.baidu.com/img/baike/new.png')}.vedio-content .drama-tab{margin:0 10px 10px;font-size:12px}.vedio-content h3{margin:0 10px 10px}.vedio-content .drama-tab a{color:#666;padding-bottom:2px;width:60px;margin-bottom:5px;display:inline-block;float:left;text-align:center;height:18px;line-height:18px;margin-left:7px}.vedio-content .drama-tab .tab-seleced{background:#47af00;color:#fff;cursor:default;width:60px}.vedio-content .drama-tab .tab-seleced:hover{text-decoration:none}.vedio-content .vedio-foot{margin:10px 10px 2px;text-align:right;color:#999;font-size:12px}.vedio-content .vedio-foot a{background:transparent url(http://img.baidu.com/img/baike/douban.gif) no-repeat scroll 0 0;margin-bottom:-3px;height:25px;width:70px;display:inline-block;vertical-align:middle}.vedio-content .vedio-foot-sp{margin-right:12px}.vedio-content .vedio-foot-sp a{background:transparent url(http://img.baidu.com/img/baike/bgs4_new.png) no-repeat scroll 0 -290px!important;height:27px;width:74px;margin-left:5px;vertical-align:baseline}.vedio-content .new-list{margin-bottom:0;float:left}.movie-content ul{margin:6px 3px 0}.movie-content .vedio-foot{margin-top:-15px}.ext-subtitle{font-family:Arial;font-size:14px;font-weight:bold;line-height:1.5;margin:10px 10px 4px}.banben-title{font-weight:bold;font-size:16px;margin-left:10px}.videospctrl{height:15px;background:#fff;margin-bottom:10px}1-1011-2021-30 第1集 第2集 第3集 第4集 第5集 第6集 第7集 第8集 第9集 第10集 "}]}
	print(concat_all(row['title'], row['summary'], row['sections']).replace(' ', '!'))
	# print(concat_all('', '', sections))

# test_tokenizer()
# test_generate()
# test_stream()
# test_json('train/zh')
# test_dataset()
# cal_model_size()
test_concat()