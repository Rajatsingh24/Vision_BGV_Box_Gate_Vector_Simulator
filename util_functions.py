import threading
import time


def suggestions(word, keys):
    def match_len(key):
        l = list(filter(lambda x: x[0] == x[1], zip(list(word)[::-1], list(key)[::-1])))
        return len(l)

    suggestions = sorted(keys, key=match_len, reverse=True)

    # suggestions = list(filter( lambda k : match_len(k)!=0 , suggestions))
    return suggestions


def naiveSuggestion(word: str, keys):
    def match_len(key):
        matched = 0
        for pair in zip(list(word), list(key)):
            if pair[0] == pair[1]:
                matched += 1
            else:
                return matched
            pass
        return matched

    suggestions = sorted(keys, key=match_len, reverse=True)
    suggestions = list(filter(lambda k: match_len(k) != 0, suggestions))
    return suggestions


def dicToStruct(dic, name):
    class struct():
        def __init__(self, dic) -> None:
            # ineranl parms
            self.thisName = name
            self.this = dic
            self.type = "dict"

            for k, v in dic.items():
                if type(k) is int:
                    self.__dict__[f"____{k}"] = v
                else:
                    self.__dict__[k] = v

    return struct(dic)


def dicRef(key: str):
    if key.count('_') == 4:
        return int(key.strip('_'))
    else:
        return key


def listToStruct(list, name):
    class struct():
        def __init__(self, dic) -> None:
            # ineranl parms
            self.thisName = name
            self.this = dic
            self.type = 'list'

            for index, v in enumerate(list):
                self.__dict__[f"index_{index}"] = v

    return struct(list)


def parser(tp, data):
    parsers = {'str': str(data), 'int': int(data), 'float': float(data), 'bool': bool(int(data))}

    try:
        return parsers[tp]
    except:
        return None


class Timer():
    def __init__(self, timerValue, trigger, onloop=False) -> None:
        self.timerValue = timerValue
        self.timer = 0
        self.trigger = trigger
        self.loop = onloop
        self.thread = threading.Thread(target=self.refreshClock)
        self.initals = None

        self.isPaused = False
        pass

    def destroy(self):
        self.thread = None
        pass

    def start(self):
        self.timer = 0
        self.initals = time.time()
        # initliing the thread
        self.thread.start()
        pass

    def pause(self):
        self.isPaused = True

        pass

    def resume(self):
        self.isPaused = False
        pass

    def reset(self):
        self.timer = 0

    def startAgain(self):
        self.resume()
        self.thread = threading.Thread(target=self.refreshClock)
        self.start()

    def refreshClock(self):
        if self.thread == None: return
        # gaurd code
        if self.isPaused: return

        self.timer = time.time() - self.initals

        if self.timer >= self.timerValue:
            # alram!!
            self.trigger()
            # to loop again
            if self.loop:
                self.initals = time.time()
                self.timer = 0
                self.thread = threading.Thread(target=self.refreshClock)
                self.thread.start()
            pass
        else:
            self.thread = threading.Thread(target=self.refreshClock)
            self.thread.start()
        pass

    pass
