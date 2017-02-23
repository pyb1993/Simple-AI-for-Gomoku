#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, time
import itertools
#----------------------------------------------------------------------
# chessboard: 棋盘类，简单从字符串加载棋局或者导出字符串，判断输赢等
#----------------------------------------------------------------------
class chessboard (object):

	def __init__ (self, forbidden = 0):
		self.__board = [ [ 0 for n in range(15) ] for m in range(15) ]
		self.__forbidden = forbidden
		self.__dirs = ( (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), \
			(1, -1), (0, -1), (-1, -1) )
		self.DIRS = self.__dirs
		self.won = {}
	
	# 清空棋盘
	def reset (self):
		for j in range(15):
			for i in range(15):
				self.__board[i][j] = 0
		return 0
	
	# 索引器->返回第i行
	def __getitem__ (self, row):
		return self.__board[row]
	

	# 将棋盘转换成字符串
	def __str__ (self):
		text = '  A B C D E F G H I J K L M N O\n'
		mark = ('. ', 'O ', 'X ')
		nrow = 0
		for row in self.__board:
			line = ''.join([ mark[n] for n in row ])
			text += chr(ord('A') + nrow) + ' ' + line
			nrow += 1
			if nrow < 15: text += '\n'
		return text
	
	# 转成字符串
	def __repr__ (self):
		return self.__str__()

	def get (self, row, col):
		if row < 0 or row >= 15 or col < 0 or col >= 15:
			return 0
		return self.__board[row][col]

	def put (self, row, col, x):
		if row >= 0 and row < 15 and col >= 0 and col < 15:
			self.__board[row][col] = x
		return 0
#----------------------------------------------------------------------
# 判断输赢，返回0（无输赢），1（白棋赢），2（黑棋赢）
# 
#----------------------------------------------------------------------
	def check (self):
		board = self.__board
		dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
		for i in range(15):
			for j in range(15):
				if board[i][j] == 0: continue
				id = board[i][j]
				for d in dirs:#8个方向开始遍历,但是只需要4个
					x, y = j, i
					count = 0
					for k in range(5):
						if self.get(y, x) != id: break
						y += d[0]
						x += d[1]
						count += 1
					if count == 5:
						self.won = {}
						r, c = i, j
						for z in range(5):
							self.won[(r, c)] = 1
							r += d[0]
							c += d[1]
						return id
		return 0
	def is_full(self):
		return len([1 for j in range(15) for i in range(15) if self.__board[i][j]!=0])==0
	
	# 返回数组对象
	def board (self):
		return self.__board
	
	# 导出棋局到字符串
	def dumps (self):
		import json
		board = self.__board
		return json.dumps(board)
	
	# 从字符串加载棋局
	def loads (self, text):
		import json
		print(text)
		self.__board = json.loads(text)
		return 0
	def load_board (self,other):
		self.__board = other
	# 输出
	def show (self):
		import os
		#os.system('cls')
		print ('  A B C D E F G H I J K L M N O')
		mark = ('. ', 'O ', 'X ')
		nrow = 0
		self.check()
		
		for row in range(15):
			print (chr(ord('A') + row),end=' ')
			for col in range(15):
				ch = self.__board[row][col]
				print (mark[ch],end='')
			print ()
		return 0


#----------------------------------------------------------------------
# evaluation: 棋盘评估类，给当前棋盘打分用
#----------------------------------------------------------------------
class evaluation (object):

	def __init__ (self):
		self.POS = []
		for i in range(15):
			row = [ (7 - max(abs(i - 7), abs(j - 7))) for j in range(15) ]#计算每个位置到边界距离的最小值
			self.POS.append(tuple(row))
		self.POS = tuple(self.POS)
		self.last = None
		self.STWO = 1		# 冲二
		self.CSTWO = 12		#连冲二
		self.CSTHREE = 13   #连冲三
		self.STHREE = 2		# 冲三
		self.SFOUR = 3		# 冲四
		self.TWO = 4		# 活二
		self.THREE = 5		# 活三
		self.FOUR = 6		# 活四
		self.FIVE = 7		# 活五
		self.NOTYPE = 11	
		self.ANALYSED = 255		# 已经分析过
		self.TODO = 0			# 没有分析过
		self.check = set((self.FIVE, self.FOUR, self.SFOUR, \
					 self.THREE, self.CSTHREE, self.STHREE, self.TWO, self.STWO))
		self.result = [ 0 for i in range(15) ]		# 保存当前直线分析值
		self.line = [ 0 for i in range(15) ]		# 当前直线数据
		self.record = []			# 全盘分析结果 [row][col][方向]
		for i in range(15):
			self.record.append([])
			self.record[i] = []
			for j in range(15):
				self.record[i].append([ 0, 0, 0, 0])
		self.count = []				# 每种棋局的个数：count[黑棋/白棋][模式]
		for i in range(3):
			data = [ 0 for j in range(20) ]
			self.count.append(data)
		self.reset()

	def reset_line(self,line):
		TODO = self.TODO
		for i,x in enumerate(line):
			line[i] = TODO
	# 复位数据 record和count
	def reset (self):
		TODO = self.TODO
		count = self.count
		for i in range(15):
			line = self.record[i]
			for j in range(15):
				line[j][0] = TODO
				line[j][1] = TODO
				line[j][2] = TODO
				line[j][3] = TODO
		for i in range(20):
			count[0][i] = 0
			count[1][i] = 0
			count[2][i] = 0
		return 0
	def find_bound(self,line,pos,num,stone):
		xl,xr = pos-1,pos+1
		while xl >= 0 and line[xl] == stone:
			xl -= 1
		while xr <= num - 1 and line[xr] == stone:
			xr += 1
		return xl,xr
	def is_continuous(self,line,pos,num,stone):
		xl,xr = self.find_bound(line,pos,num,stone)
		return xl>=0 and xr < num and line[xl] == 0 and line[xr] == 0#两边都是通的
	# 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
	def evaluate (self, board, turn):
		score = self.__evaluate(board, turn)
		count = self.count
		return score
	
	def analysis_result(self,board,turn):
		self.count_state(board)
		return self._analysis_result(board,turn)
	
	def count_state(self,board):
		# 分别对白棋黑棋计算：FIVE, FOUR, THREE, TWO等出现的次数
		check = self.check
		record,count = self.record,self.count
		for i,row in enumerate(board):
			for j,stone in enumerate(row):
				if stone == 0: continue
				for k in range(4):
					ch = record[i][j][k]#获取每个位置的状态
					if ch in check:
						count[stone][ch] += 1#统计每个位置状态的数量
		
	def _analysis_result(self,board,turn):
		FIVE, FOUR, THREE, TWO = self.FIVE, self.FOUR, self.THREE, self.TWO
		SFOUR, CSTHREE,STHREE, STWO = self.SFOUR, self.CSTHREE, self.STHREE, self.STWO
		count = self.count
		# 如果有五连则马上返回分数

		inverse = 1 if turn == 2 else 2
		if count[inverse][FIVE]:
			return -100000
		elif count[turn][FIVE]:
			return 100000
		
		# 如果存在两个冲四，则相当于有一个活四
		if count[turn][SFOUR] >= 2:
			count[turn][FOUR] += 1
		if count[inverse][SFOUR] >= 2:
			count[inverse][FOUR] += 1

		# 具体打分
		wvalue, bvalue, win = 0, 0, 0
		#x下面这一部分是马上要返回的,因为必须马上防守或者对方已经必死了
		
		if count[turn][FOUR] > 0:return 9990#我方存在一个活四 ****(肯定赢了)
		if count[turn][SFOUR] > 0: return 9980#我方存在一个冲四 *** *(肯定赢了)
	#以下我方不存在冲四和活四
		if count[inverse][FOUR] > 0: return -9970#对方存在一个活四(肯定输了)
	#以下对方不存在活四
		if count[inverse][SFOUR] and count[inverse][THREE]: #对方存在冲四以及活三(肯定输了)
			return -9960
		if count[turn][THREE] and count[inverse][SFOUR] == 0:#我方存在活三以及对方不存在冲四(肯定赢了)
			return 9950
		#对方存在两个活三，我方不存在活三,冲三(我方基本必死)
		if	count[inverse][THREE] > 1 and \
			count[turn][THREE] == 0 and \
			count[turn][STHREE] == 0:
				return -9940

	#下面的情况不是马上返回的状态
		if count[turn][THREE]:#我方存在1个或者以上的活三(基本赢了,对方可以有冲四来缓解)
			wvalue += 2000 if count[turn][THREE]>1 else 200
		if count[inverse][THREE]:#对方存在一个以上的活三(因为我方占先手,可以选择堵住,所以加分少)
			bvalue += 500 if count[inverse][THREE]>1 else 100
		if count[turn][CSTHREE]:#我方存在一个以上的连冲三
			wvalue += 1500 if count[turn][THREE]>1 else 150
		if count[inverse][CSTHREE]:#对方存在一个以上的活三(因为我方占先手,可以选择堵住,所以加分少)
			bvalue += 400 if count[inverse][THREE]>1 else 80
		if count[turn][STHREE]:#我方存在一个以上的冲三
			wvalue += count[turn][STHREE] * 10
		if count[inverse][STHREE]:#对方存在一个以上的冲三
			bvalue += count[inverse][STHREE] * 10
		if count[turn][TWO]:#活二
			wvalue += count[turn][TWO] * 4
		if count[inverse][TWO]:
			bvalue += count[inverse][TWO] * 4
		if count[turn][STWO]:#我方存在一个冲二
			wvalue += count[turn][STWO]
		if count[inverse][STWO]:
			bvalue += count[inverse][STWO]

		# 加上位置权值，棋盘最中心点权值是7，往外一格-1，最外圈是0
		#在中心的位置具有更好的地里位置
		wc, bc = 0, 0
		for i in range(15):
			for j in range(15):
				stone = board[i][j]
				if stone == 0: continue
				if stone == turn:
					wc += self.POS[i][j]
				else:
					bc += self.POS[i][j]
		wvalue += wc
		bvalue += bc
		
		return wvalue - bvalue
	
	# 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
	def __evaluate (self, board, turn):
		record, count = self.record, self.count
		TODO, ANALYSED = self.TODO, self.ANALYSED
		self.reset()
		# 四个方向分析
		for i in range(15):
			boardrow = board[i]#实际数据
			recordrow = record[i]#分析结果
			for j in range(15):#针对每一个点分析可能的走法和分数
				if boardrow[j] != 0:
					if recordrow[j][0] == TODO:		# 水平没有分析过？
						self.__analysis_horizon(board, i, j)
					if recordrow[j][1] == TODO:		# 垂直没有分析过？
						self.__analysis_vertical(board, i, j)
					if recordrow[j][2] == TODO:		# 左斜没有分析过？
						self.__analysis_left(board, i, j)
					if recordrow[j][3] == TODO:		# 右斜没有分析过
						self.__analysis_right(board, i, j)
		return self.analysis_result(board,turn)
		

	# 分析横向,result 记录某一行的结果
	def __analysis_horizon (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		line = board[i].copy()
		self.reset_line(result)
		self.analysis_line(line, result, 15, j)
		for x in range(15):
			if result[x] != TODO:
				record[i][x][0] = result[x]
		return record[i][j][0]
	
	# 分析横向
	#这里利用record来记录这个点在这个方向是不是计算过
	def __analysis_vertical (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		line = [board[x][j] for x in range(15)]#获取一列
		self.reset_line(result)
		self.analysis_line(line, result, 15, i)
		for x in range(15):
			if result[x] != TODO:
				record[x][j][1] = result[x]
		return record[i][j][1]
	
	# 分析左斜
	def __analysis_left (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		if i < j: x, y = j - i, 0
		else: x, y = 0, i - j
		self.reset_line(result)
		realnum = 15-max(x,y)#计算边界
		line = [board[y+k][x+k] for k in range(realnum)]
		self.analysis_line(line, result, realnum, j - x)
		for s in range(realnum):
			if result[s] != TODO:
				record[y + s][x + s][2] = result[s]
		return record[i][j][2]

	# 分析右斜
	def __analysis_right (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		if 14 - i < j: x, y = j - 14 + i, 14#下半部分
		else: 		   x, y = 0, i + j#上半部分
		self.reset_line(result)
		realnum = min(y,14-x)+1
		line = [board[y-k][x+k] for k in range(realnum)]
		self.analysis_line(line, result, realnum, j - x)
		for s in range(realnum):
			if result[s] != TODO:
				record[y - s][x + s][3] = result[s]
		return record[i][j][3]
	

	# 分析一条线：五四三二等棋型
	# pos 代表是这条线上面的第几个点
	def analysis_line (self, line, record, num, pos):
		if line[pos] == 0:
			return
		TODO, ANALYSED = self.TODO, self.ANALYSED
		TWO,STWO = self.TWO,self.STWO
		THREE, STHREE = self.THREE, self.STHREE
		FOUR, SFOUR = self.FOUR, self.SFOUR
		
		if num < 5:
			for i in range(num): 
				record[i] = ANALYSED
			return 0
			
		stone = line[pos]#当前棋子的颜色
		inverse = (0, 2, 1)[stone]#黑的变成白,白的变成黑的
		
		xl,xr = self.find_bound(line,pos,pos,stone)#寻找属于自己的范围(不含空格)
		xl += 1
		xr -= 1

		left_range = xl
		right_range = xr
		while left_range > 0 and line[left_range-1]!=inverse:	# 探索左边范围（非对方棋子的格子坐标）
			left_range -= 1
		while right_range < num-1 and line[right_range+1]!=inverse:	# 探索右边范围（非对方棋子的格子坐标）
			right_range += 1

		# 如果该直线范围小于 5，则直接返回(不可能连成5个)
		if right_range - left_range < 4:
			for k in range(left_range, right_range + 1):
				record[k] = ANALYSED
			return 0

		# 设置已经分析过
		for k in range(xl, xr + 1):
			record[k] = ANALYSED

		srange = xr - xl

		# 如果是 5连
		if srange >= 4:	
			record[pos] = self.FIVE
			return self.FIVE
		
		# 如果是 4连
		if srange == 3:	
			leftfour = False	# 是否左边是空格
			rightfour = False
			if xl > 0:
				leftfour = (line[xl - 1] == 0)
			if xr < num-1:
				rightfour = (line[xr + 1] == 0)

			record[pos] = (ANALYSED,SFOUR,FOUR)[leftfour+rightfour]
			return record[pos]
		
		# 如果是 3连
		if srange == 2:		# 三连
			left3 = False	# 是否左边是空格
			right3 = False
			if xl > 0 and line[xl-1]==0:
				left3 = True
				if xl > 1 and line[xl - 2] == stone:
					record[xl] = SFOUR
					record[xl - 2] = ANALYSED
					return

			if xr < num-1 and line[xr+1]==0:
				right3 = True
				
				if xr < num - 2 and line[xr + 2] == stone:
					record[xr] = SFOUR	# XXX-X 相当于冲四
					record[xr + 2] = ANALYSED
					return
			record[pos] = (ANALYSED,STHREE,THREE)[left3+right3]
			
			return record[pos]
		
		# 如果是 2连
		if srange == 1:		# 两连
			left2 = False
			right2 = False

			if xl > 0 and line[xl-1]==0:
				left2 = True
				#if pos == 7:print('weird',xl,xr)
				if xl>1 and line[xl-2] == stone:#至少冲三
					if xl>2 and line[xl-3] == stone:#冲四
						record[xl - 3] = ANALYSED
						record[xl - 2] = ANALYSED
						record[xl]	   = SFOUR
						return SFOUR
					else:#冲三不能冲四
						record[xl - 2] = ANALYSED
						record[xl] 	   = STHREE
						if self.is_continuous(line,xl-1,num,stone):
							record[xl] = self.CSTHREE#连冲三
						return STHREE
		
			if xr < num - 1 and line[xr + 1]==0:
				right2 = True

				if xr < num -2 and line[xr + 2]==stone:#至少冲三
					if xr < num - 3 and line[xr + 3]==stone:#冲四
						record[xr + 3] = ANALYSED
						record[xr + 2] = ANALYSED
						record[xr] 	   = SFOUR
						return SFOUR
					else:
						record[xr+2] = ANALYSED
						record[xr] 	 = STHREE
						if self.is_continuous(line,xr+1,num,stone):
							record[xr] = self.CSTHREE
						return STHREE
			record[pos] = (ANALYSED,STWO,TWO)[left2+right2]
			
			return record[pos]
		return 0

#----------------------------------------------------------------------
# DFS: 博弈树搜索
#----------------------------------------------------------------------
class searcher (object):

	# 初始化
	def __init__ (self,maxdepth = 3,maxwidth = 15*15):
		from collections import deque
		self.evaluator = evaluation()
		self.board = [ [ 0 for n in range(15) ] for i in range(15) ]
		self.overvalue = 0
		self.maxwidth = maxwidth
		self.maxdepth = maxdepth
		self.topturn = 0
		self.path = deque()
	def set_turn(self,topturn):
		self.topturn = topturn
	# 产生当前棋局的走法
	def genmove (self, turn):
		moves = []
		board = self.board
		POSES = self.evaluator.POS
		inverse = 1 if turn==2 else 2
		for i in range(15):
			for j in range(15):
				if board[i][j] == 0:
				
					board[i][j] = turn
					score = -self.evaluator.evaluate(board,inverse)#获取每一个位置的分数
					board[i][j] = 0
					moves.append((score, i, j))
		moves.sort(reverse=True)#从大到小排序
		
		return moves[:self.maxwidth]
	
	# 递归搜索：返回最佳分数
	#最大最小剪枝
	def __search (self, turn, depth,current_score = -10000,alpha = -10000, beta = 10000):

		# 深度为零则评估棋盘并返回
		if depth <= 0:
			return current_score
		
		#游戏结束,返回
		if (abs(current_score) >= 9000) and depth < self.maxdepth:
			return current_score

		# 产生新的走法
		possible_moves = self.genmove(turn)
		moves = list(filter(lambda x : x[0] > -9000,possible_moves)).copy()#过滤掉必输的走法
		bestmove = None

		#考虑有没有直接获胜的走法,有的话跳过后续检查
		# if depth == self.maxdepth:
			# if len(moves) > 0 and moves[0][0] > 9000:#几乎必胜
				# score,row,col = moves[0]
				# self.bestmove = (row,col)
				# return score

		# 枚举当前所有走法
		for next_score, row, col in moves:
			# 标记当前走法到棋盘
			if next_score > 9000 and depth == self.maxdepth:
				self.bestmove = (row,col)
				print('FFFFFFFFFFFFFFFFFFFFFFFFFFFF',next_score,self.bestmove)
				return next_score
			self.board[row][col] = turn
			nturn = 2 if turn == 1 else 1
			score = - self.__search(nturn, depth - 1,-next_score,-beta, -alpha)#注意负号,对方的最好在我方就是最不好
			self.board[row][col] = 0

			# 计算最好分值的走法
			# alpha/beta 剪枝
			if score > alpha:
				#if score > 9000 and turn == topturn:print(score)
				alpha = score
				bestmove = (row, col)
				if alpha >= beta:
					break
			if depth == self.maxdepth and score >= 9000:#找到一种获胜的走法
				bestmove = (row,col)
				break
		# 如果是第一层则记录最好的走法
		if depth == self.maxdepth:
			
			if bestmove == None:
				bestmove = possible_moves[0][1:]
			self.bestmove = bestmove
			
			assert(self.bestmove!=None)
				#print(possible_moves)
				#assert(1==0)
		# 返回当前最好的分数，和该分数的对应走法
		return alpha

	# 具体搜索：传入当前是该谁走(turn=1/2)，以及搜索深度(depth)
	def search (self, turn):
		self.bestmove = None
		score = self.__search(turn, self.maxdepth)
		row, col = self.bestmove
		return score, row, col

class advanced_searcher(searcher):
	def __init__(self,maxdepth = 1,maxwidth = 50):
		super().__init__(maxdepth = maxdepth,maxwidth = maxwidth)
		self.evaluator = efficient_evaluator()
	def dfs(self,turn,begin,pos,vis):
		
		board = self.board
		vis = set()
		dir = ((0,1),(1,0),(1,1),(-1,-1),(1,-1),(0,-1),(-1,0),(-1,1))
		inverse = 1 if turn == 2 else 2
		t_num,i_num = 0, 0
		
		def _dfs(i,j):
			nonlocal vis,board,dir,inverse,turn,t_num,i_num
			ch = board[i][j]
			if ch == inverse:
				i_num += 1
				return
			t_num += 1
		
			for dx,dy in dir:
				nx,ny = i + dx, j + dy
				if board[nx][ny]!=0:
					if (nx,ny) not in vis:
						vis.add((nx , ny))
						_dfs(nx,ny)
				else:
					pos.append((nx,ny))#可以重复添加空位置 
	
		_dfs(begin[0],begin[1])

		return t_num + 0.2*i_num#返回这一片连通域的打分

	def pre_genmove(self,turn):
		from collections import defaultdict
		moves = []
		board = self.board
		POSES = self.evaluator.POS
		inverse = 1 if turn==2 else 2
		vis = {}
		grades = defaultdict(lambda : 0)
		for i in range(15):
			for j in range(15):
				if board[i][j] != 0 and (i,j) not in vis:
					pos = []
					score = self.dfs(board[i][j],(i,j),pos,vis)
					
					for each in pos:
						grades[each] += 0.2*score + 1# +1代表有一个turn棋子相邻, 0.2*score代表这个连通量的分数
		
		for p,score in grades.items():
			i,j = p
			moves.append((POSES[i][j] + score,i,j))
		moves.sort(reverse=True)#从大到小排序
		
		#return sorted([(100,i,j) for i in range (15) for j in range(15) if board[i][j]==0],reverse=True)[:50]
		return moves[:50]

	def genmove(self,turn):
		moves = []
		board = self.board
		POSES = self.evaluator.POS
		inverse = 1 if turn==2 else 2
		e = self.evaluator
		#----------------------------#
		e.reset()#清空所有的状态
		e.evaluate(board,turn)#构造状态
		#----------------------------#
		last = None
		pre_moves = self.pre_genmove(turn)
		for score,i,j in pre_moves:
				if board[i][j] != 0: continue
				if last != None:
					e.undo_point(last[0],last[1],board)#撤销上一个点带来的影响

				#下面增加一个点,只计算一个点的影响
				board[i][j] = turn
				e.add_point(i,j,board)
				last = (i,j)
				score = - e.analysis_result(board,inverse)#获取每一个位置的分数
				board[i][j] = 0
				#yield (score,i,j)
				moves.append((score, i, j))

		moves.sort(reverse=True)#从大到小排序
		return moves[:self.maxwidth]
class efficient_evaluator(evaluation):
	def reset_count(self):
		for i in range(3):
			for j in range(20):
				self.count[i][j] = 0
	def add_point(self,i,j,board):
		self.reset_count()#置空
		if 0:
			record = self.record
			for x in range(15):
				for y in range(15):
					for k in range(4):
						t = record[x][y][k]
						if t == 3 and board[x][y]==2:
							print('^^^',x,y,k,'SFOUR')
		self.modify_point_horizion(i,j,board)
		self.modify_point_verticle(i,j,board)
		self.modify_point_left(i,j,board)
		self.modify_point_right(i,j,board)
		if 0:
			record = self.record
			for x in range(15):
				for y in range(15):
					for k in range(4):
						t = record[x][y][k]
						if t == 3 and board[x][y]==2:
							print('---',x,y,k,'SFOUR')
	def modify_point_horizion(self,i,j,board):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		result = self.result
		self.reset_line(result)# set TODO

		line = board[i].copy()
		for p in range(15):
			if line[p] and result[p] == TODO:
				x = self.analysis_line(line,result,15,p)

		for x in range(15):
			record[i][x][0] = result[x]
		#print(result,'horizon\n',line)
	def modify_point_verticle(self,i,j,board):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		
		line = [board[x][j] for x in range(15)]#获取一列
		self.reset_line(result)# set TODO

		for p in range(15):
			if line[p] and result[p] == TODO:
				self.analysis_line(line,result,15,p)
		
		for x in range(15):
			record[x][j][1] = result[x]

	def modify_point_left(self,i,j,board):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		
		self.reset_line(result)# set TODO
			
		if i < j: x, y = j - i, 0
		else: x, y = 0, i - j

		realnum = 15-max(x,y)#计算边界
		line = [board[y+k][x+k] for k in range(realnum)]

		for p in range(realnum):
			if line[p] and result[p] == TODO:
				self.analysis_line(line,result,realnum,p)
		
		for s in range(realnum):
			record[y + s][x + s][2] = result[s]
		#print(result,'left')
	def modify_point_right(self,i,j,board):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		self.reset_line(result)# set TODO

		if i + j > 14: x, y = j - 14 + i, 14#下半部分
		else: 		   x, y = 0, i + j#上半部分
		
		realnum = min(y,14-x)+1
		line = [board[y-k][x+k] for k in range(realnum)]

		for p in range(realnum):
			if line[p] and result[p] == TODO:
				self.analysis_line(line,result,realnum,p)
		
		for s in range(realnum):
			record[y - s][x + s][3] = result[s]

	def undo_point(self,i,j,board):
		self.modify_point_horizion(i,j,board)
		self.modify_point_verticle(i,j,board)
		self.modify_point_left(i,j,board)
		self.modify_point_right(i,j,board)

#----------------------------------------------------------------------
# psyco speedup
#----------------------------------------------------------------------
def psyco_speedup ():
	try:
		import psyco
		psyco.bind(chessboard)
		psyco.bind(evaluation)
	except:
		pass
	return 0

psyco_speedup()


#----------------------------------------------------------+------------
# main game
#----------------------------------------------------------------------
def gamemain():
	b = chessboard()
	opening = [[0]*15 for i in range(15)]
	opening[7][7] = 2
	opening[7][8] = 1
	b.load_board(opening)
	turn = 2
	history = []
	undo = False
	s = searcher()
	s.board = b.board()
	# 设置难度
	DEPTH = 1

	if len(sys.argv) > 1:
		if sys.argv[1].lower() == 'hard':
			DEPTH = 2

	while 1:
		print ('')
		while 1:
			print ('<ROUND %d>'%(len(history) + 1))
			b.show()
			print ('Your move (u:undo, q:quit):',)
			text = input().strip('\r\n\t ')
			if len(text) == 2:
				tr = ord(text[0].upper()) - ord('A')
				tc = ord(text[1].upper()) - ord('A')
				if tr >= 0 and tc >= 0 and tr < 15 and tc < 15:
					if b[tr][tc] == 0:
						row, col = tr, tc
						break
					else:
						print ('can not move there')
				else:
					print ('bad position')
			elif text.upper() == 'U':
				undo = True
				break
			elif text.upper() == 'Q':
				print (b.dumps())
				return 0
		
		if undo == True:
			undo = False
			if len(history) == 0:
				print ('no history to undo')
			else:
				print ('rollback from history ...')
				move = history.pop()
				b.loads(move)
		else:
			history.append(b.dumps())
			b[row][col] = 1

			if b.check() == 1:
				b.show()
				print (b.dumps())
				print ('YOU WIN !!')
				return 0

			print ('robot is thinking now ...')
			score, row, col = s.search(2, DEPTH)
			cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
			print ('robot move to %s (%d)'%(cord, score))
			b[row][col] = 2

			if b.check() == 2:
				b.show()
				print (b.dumps())
				print ('')
				print ('YOU LOSE.')
				return 0

	return 0


#----------------------------------------------------------------------
# testing case
#----------------------------------------------------------------------
if __name__ == '__main__':
	def test1():
		b = chessboard()
		b[10][10] = 1
		b[11][11] = 2
		for i in range(4):
			b[5 + i][2 + i] = 2
		for i in range(4):
			b[7 - 0][3 + i] = 2
		print (b)
		print ('check', b.check())
		return 0

	def test2():
		b = chessboard()
		b[7][7] = 1
		b[8][8] = 2
		b[7][9] = 1
		eva = evaluation()
		for l in eva.POS: print (l)
		return 0
	def test3():
		e = evaluation()
		line = [ 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
		record = [e.TODO]*10
		e.analysis_line(line, record, len(line), 6)
		assert (record[:10]==[0,0,255,0,3,255,255,0,0,0])
		return 0
	def test4():
		from itertools import chain
		WHITE,BLACK = 1 , 2
		b = chessboard()
		eva = evaluation()
		THREE,SFOUR = eva.THREE,eva.SFOUR
		board = [[0]*15 for i in range(15)]

		
		board[1] = [1,0,1,1,1,2,0,0,0,0,0,0,0,0,0,0]
		board[2] = [2,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0]
		b.load_board(board)
		t = time.time()
		for i in range(1000):
			score = eva.evaluate(b.board(), 2)
		assert(eva.count[WHITE][SFOUR]==1)
		assert(eva.count[BLACK][THREE]==1)
		t = time.time() - t
		return 0
	def test5(name):
		import profile
		profile.run("%s"%(name), "prof.txt")
		import pstats
		p = pstats.Stats("prof.txt")
		p.sort_stats("time").print_stats()
	def test6():
		from itertools import chain
		b = chessboard()
		#b.loads('1:CJ 2:DJ 1:dk 1:DL 1:EH 1:EI 2:EJ 2:EK 2:FH 2:FI 2:FJ 1:FK 2:FL 1:FM 2:GF 1:GG 2:GH 2:GI 2:GJ 1:GK 1:GL 2:GM 1:HE 2:HF 2:HG 2:HH 2:HI 1:HJ 2:HK 2:HL 1:IF 1:IG 1:IH 2:II 1:IJ 2:IL 2:JG 1:JH 1:JI 1:JJ 1:JK 2:JL 1:JM 1:KI 2:KJ 1:KL 1:LJ 2:MK')
		#b.loads('1:HH,1:HI,1:HJ,1:HK')
		a1 = {'DF':2,'EG':1,'FG':1,'FH':1,'FJ':2,'GG':2,'GH':1,'GI':1,'HG':2,'HH':1,'IG':1,'IH':2,'JF':1,'JI':2,'KE':1}
		a2 = {'CE':2 ,'CK':2 ,'DF':1, 'DK':1 ,'DL':2, 'EG':1, 'EI':1, 'EK':1,\
			'FG':2 ,'FH':1, 'FI':1 ,'FJ':1 ,'FK':1, 'FL':2 ,'GD':1, 'GE':2,\
			'GF':2,'GG':2, 'GH':1 ,'GI':1, 'GK':1,'HG':2 ,'HH':1 ,'HJ':2,'HK':2,'IG':2,'JG':1,'AA':2}
		for key,val in chain(a1.items(),a2.items()):
			r = ord(key[0])-ord('A')
			c = ord(key[1])-ord('A')
			b[r][c] = val
		s = searcher()
		s.board = b.board()
		t = time.time()
		#b.show()
		score, row, col = s.search(2, 2)
		t = time.time() - t
		b[row][col] = 2
		
		assert((row,col) == (10,9))

	def test7(b,s1,s2,record,who1,who2):#左右互博
		s1.set_turn(1)
		s2.set_turn(2)
		step = 0
		while step < 15*15:

			score, row, col = s1.search(1)
			b[row][col] = 1
			cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
			print ('robot move to %s (%d)'%(cord, score))
			b.show()
			step += 1
			if b.check() == 1:
				print(step)
				record[who1]+= (255-step)/255*10
				return 0

			# s2 think
			score, row, col = s2.search(2)
			cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
			print ('robot move to %s (%d)'%(cord, score))
			
			step += 1
			b[row][col] = 2
			b.show()
			
			if b.check() == 2:
				print(step)
				record[who2]+= (255-step)/255*10
				return 0
		return 0

	
	def test8(seacher_cls):
		print('start')
		record = {1:0,2:0}
		for i in range(3):
			b = chessboard()
			# 1 50, 3 50
			s1 = searcher(maxdepth = 1,maxwidth = 225)
			s2 = seacher_cls(maxdepth = 4,maxwidth = 10)
			opening = [[0]*15 for i in range(15)]
			opening[7][7] = 2
			opening[7][8] = 1
			b.load_board(opening)
			s1.board = b.board()
			s2.board = b.board()
			if(i%2==0):
				test7(b,s1,s2,record,1,2)#s1先走
			else:
				test7(b,s2,s1,record,2,1)#s2先走
			print(record)
			a = input()
			
	def test9(which):
		for u in range(1):
			board = [[] for i in range(15)]
			R1 = {}
			R2 = {}
			#test CSTHREE
			# board[6] = [0,0,0,0,0,2,0,2,2]
			# board[7] = [0,0,0,0,0,0,1,2,1]
			# board[8] = [0,0,0,0,0,0,2,1,1]
			# board[9] = [0,0,0,0,0,1,2,0,1]
			i = 0
			#board[7] = [0,0,0,0,0,1,0,0,0,0]
			with open('fuckboard1.txt') as f:
				for line in f:
					for x in line:
						if x == '.':board[i].append(0)
						elif x== 'O':board[i].append(1)
						elif x == 'X':board[i].append(2)
					i += 1
			# for i in range(15):
				# while len(board[i])<15:
					# board[i].append(0)
			b = chessboard()
			b.load_board(board)
			b.show()
			s =searcher(maxdepth = 1,maxwidth = 225)
			s.set_turn(2)
			s.board = b.board()
			s2 = advanced_searcher(maxdepth = 1,maxwidth = 225)
			s2.board = b.board()
			s2.set_turn(2)

			for i in s.genmove(2):
				score,row,col = i
				cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
				print ('robot move to %s (%d)'%(cord, score),row,col)
				R1[cord] = score
			for i in s2.genmove(2):
				score,row,col = i
				cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
				print ('robot move to %s (%d)'%(cord, score),row,col)
				R2[cord] = score
				if cord in R1 and abs(R1[cord]-R2[cord]) > 0:
					print('ssssss',cord,R1[cord],R2[cord])

					#c,c2 = s.evaluator.count,s2.evaluator.count
					#e = s.evaluator
					# for i in range(1,3):
							# for k in range(20):
								# x,y = c[i][k],c2[i][k]
								# if x!=y:
									# print(('white','black')[i-1],'shape:',k,'num1:',x,'num2:',y)
				
	def test10():
		board = [[] for i in range(15)]
		#test CSTHREE
		# board[6] = [0,0,0,0,0,2,0,2,2]
		# board[7] = [0,0,0,0,0,0,1,2,1]
		# board[8] = [0,0,0,0,0,0,2,1,1]
		# board[9] = [0,0,0,0,0,1,2,0,1]
		i = 0
		#board[7] = [0,0,0,0,0,1,0,0,0,0]
		with open('fuckboard1.txt') as f:
			for line in f:
				for x in line:
					if x == '.':board[i].append(0)
					elif x== 'O':board[i].append(1)
					elif x == 'X':board[i].append(2)
				i += 1
		b = chessboard()
		b.load_board(board)
		s = advanced_searcher(maxdepth = 1,maxwidth = 50)
		s.board = b.board()
		s.set_turn(2)
		m = s.pre_genmove(2)
		b.show()
		for score,row,col in m:
 
			cord = '%s%s'%(chr(ord('A') + row), chr(ord('A') + col))
			print ('robot move to %s (%d)'%(cord, score),row,col)

	#test8(advanced_searcher)
	#gamemain()
	#test9(1)
	#test5('test9(2)')
	#print('advanced_seacher')
	#test5('test8(advanced_searcher)')
	#test5('advanced_searcher')
	test10()

