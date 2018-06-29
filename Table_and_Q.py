import numpy as np
import pandas as pd
from random import randint,random
from collections import deque
from time import sleep
from IPython.display import clear_output,Markdown,display
from ipywidgets import Layout,Box,Button
import matplotlib.pyplot as plt

def sample_table():
    contents = np.random.random((3,5))
    index = ['choose_a','choose_b','choose_c']
    put_state_name = np.vectorize(lambda x:'state_'+x)
    columns = put_state_name(np.random.randint(low=111,high=999,size=5).astype('str')).tolist()
    a_table = pd.DataFrame(contents,index=index,columns=columns)
    return a_table
def heads_of_three():
    return randint(1,9),randint(1,9),randint(1,9)
def get_table():
    return pd.DataFrame(np.zeros(9).reshape(3,3),index=['a','b','c'],columns=['left','middle','right'])
def next_machine(abc):
    if abc%3==0:
        hint='b'
    elif abc%2==0:
        hint='a'
    else:
        hint='c'
    return hint
def create_table(table,a,b,c,hint):
    table.left.iloc[0]=a
    table.left.iloc[1]=b
    table.left.iloc[2]=c
    
    table.middle.loc[hint]=table.left.loc[hint]
    table.right.loc[hint]=table.left.loc[hint]
    
    if hint:
        for idx in ['a','b','c']:
            if idx!=hint:
                table.middle.loc[idx]=randint(1,9)
                table.right.loc[idx]=randint(1,9)
    else:
        for idx in ['a','b','c']:
            table.middle.loc[idx]=randint(1,9)
            table.right.loc[idx]=randint(1,9)
    return table

class Table(object):
    def __init__(self,first_col=True,networth=1000):
        super(Table,self).__init__()
        self.first_col = first_col
        self.hints = deque([np.random.choice(['a','b','c']),np.random.choice(['a','b','c'])],maxlen=2)
        self.hints2 = deque([np.random.choice(['a','b','c']),np.random.choice(['a','b','c'])],maxlen=2)
        self.networth = networth
        self.reward = 0
        a,b,c = heads_of_three()
        table = get_table()
        if first_col:
            self.table = create_table(table,a,b,c,hint=self.hints[-1])
        else:
            self.table = create_table(table,a,b,c,hint=self.hints2[-1])
    def update_hints(self):
        sum_left = self.table.left.sum()
        sum_whole = self.table.values.sum()
        #print(sum_left,sum_whole)
        self.hints.append(next_machine(sum_left))
        self.hints2.append(next_machine(sum_whole))
    
    def update_table(self):
        self.update_hints()
        a,b,c = heads_of_three()
        #pirnt(hints2)
        if self.first_col:
            self.table = create_table(self.table,a,b,c,self.hints[-1])
        else:
            self.table = create_table(self.table,a,b,c,self.hints2[-1])
        #return self.table
    
    def get_score_and_update_networth(self,bet_on_machine):
        self.update_table()
        tmp = self.table.loc[bet_on_machine]
        if np.mean(tmp)==tmp.iloc[0]:
            #reword = pd.Series(np.sum(tmp))
            self.reward = np.sum(tmp)
            self.networth += self.reward
            '''
            networth = pd.Series(self.networth)
            table = pd.concat([reword,tmp],axis=0)
            table.rename(index={0:'reword'},inplace=True)
            table = pd.concat([table,networth],axis=0).rename(index={0:'NetWorth'})
            '''
            print('you win {} this time. your current networth is {}'.
                  format(self.reward,self.networth))
            print(self.table)
            print('player has choosen the right machine.')
            #return self.table #.transpose()
        else:
            #reword = pd.Series(-10)
            self.reward = -50
            self.networth += self.reward
            '''
            networth = pd.Series(self.networth)
            table = pd.concat([reword,tmp],axis=0)
            table.rename(index={0:'reword'},inplace=True)
            table = pd.concat([table,networth],axis=0).rename(index={0:'NetWorth'})
            '''
            print('you win {} this time. your current networth is {}'.
                  format(self.reward,self.networth))
            print(self.table)
            print('player has choosen a wrong machine.')
            if self.networth<0:
                print('没钱了亲，贷款请联系微信：123456789（很方便的高利贷）。')
            #return self.table #.transpose()
        
    def get_env_and_reward(self):
        env = ''
        table = self.table.transpose().values.reshape(-1).tolist()
        for v in table:
            env += str(int(v))
        return env[:3],self.reward
    
def pull_the_plug(machine):
    table.get_score_and_update_networth(machine)
table = Table()
box_layout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    #border='solid',
                    width='50%')
button_a = Button(description="a",tooltip='tooltip',icon='icon')
button_a.style.button_color = 'lightgreen'
button_b = Button(description='b',tooltip='tooltip',icon='icon')
button_b.style.button_color = 'lightgreen'
button_c = Button(description='c',tooltip='tooltip',icon='icon')
button_c.style.button_color = 'lightgreen'
buttons = [button_a,button_b,button_c]
box = Box(children=buttons,layout=box_layout)
def pull_a(a):
    clear_output()
    pull_the_plug('a')
    display(box)
def pull_b(a):
    clear_output()
    pull_the_plug('b')
    display(box)
def pull_c(a):
    clear_output()
    pull_the_plug('c')
    display(box)

class Q_learning(object):
    def __init__(self,learning_rate,epsilon,networth):
        super(Q_learning,self).__init__()
        self.Q_table = {}
        self.learning_rate = learning_rate
        self.table = Table()
        self.epsilon = epsilon
        self.og_epsilon = epsilon
        self.networth = networth
        self.graduate = False
        self.networth_hist = []
        self.trail_count = 0
    def init_agent(self):
        self.networth_hist.append(self.networth)
        self.trail_count=0
        self.table = Table()
        self.networth=1000
        print('initiating agent')
    def update_q_table(self,env,action,reward):
        if env in self.Q_table.keys():
            if action in self.Q_table[env].keys():
                self.Q_table[env][action] += (1-self.learning_rate)*\
                self.Q_table[env][action]+self.learning_rate*reward
            else:
                self.Q_table[env][action]=self.learning_rate*reward
        else:
            self.Q_table[env] = {}
            self.Q_table[env][action] = 0
    def choose_action(self,env):
        #if self.Q_table[env]:
        if env in self.Q_table.keys():
            action = max(self.Q_table[env],key=self.Q_table[env].get)
        else:
            action = np.random.choice(['a','b','c'])
        return action
    def choose_random_action(self):
        return np.random.choice(['a','b','c'])
    def epsilon_decay(self):
        self.epsilon = 0.9999*self.epsilon
    def run_one_game(self,explore=True,sleep_time=None):
        if sleep_time:
            sleep(sleep_time)
        if explore: #explore is self.epsilon>25
            if (self.networth>0) & (self.trail_count<40):
                clear_output()
                action=self.choose_random_action()
                env,_ = self.table.get_env_and_reward()
                self.table.get_score_and_update_networth(action)
                _,reward = self.table.get_env_and_reward()
                self.update_q_table(env,action,reward)
                self.epsilon_decay()
                self.networth = self.table.networth
                self.trail_count+=1
            else:
                self.init_agent()
                self.networth = self.table.networth
        else:
            if (self.networth>0) & (self.trail_count<40):
                clear_output()
                env,_ = self.table.get_env_and_reward()
                action = self.choose_action(env)
                self.table.get_score_and_update_networth(action)
                _,reward = self.table.get_env_and_reward()
                self.update_q_table(env,action,reward)
                self.epsilon_decay()
                self.networth = self.table.networth
                if self.networth>=10000:
                    self.graduate=True
                    return
            else:
                self.init_agent()
                self.networth = self.table.networth
    def train(self,sleep_time=None):
            for i in range(20000):
                display(Markdown('training'))
                too_early = self.epsilon>25 # 25 is the tolerance of training iteration that is allowed.
                if too_early:
                    explore = random()<(self.epsilon/self.og_epsilon) # solution for E&E trade-off
                else:
                    explore=False
                self.run_one_game(explore,sleep_time=sleep_time)
                if self.graduate:
                    print('the agent is too good to play in this game')
                    return
    def idiot_demo(self,demos=30,sleep_time=1.5):
        self.init_agent()
        for i in range(demos):
            display(Markdown('idiot is playing'))
            self.run_one_game(explore=True,sleep_time=sleep_time)
    def master_demo(self,demos=50,sleep_time=1.5):
        self.init_agent()
        for i in range(demos):
            display(Markdown('master is playing'))
            self.run_one_game(explore=False,sleep_time=sleep_time)
    def draw_networth_hist(self):
        x_line = np.arange(len(self.networth_hist)-1)
        plt.plot(x_line,self.networth_hist[1:])
        plt.title('networth over trails (leftover money after 40 times of playing)')
        plt.xlabel('trails')
        plt.ylabel('networth')
        plt.show();