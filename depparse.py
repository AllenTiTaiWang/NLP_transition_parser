from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence, Text, Union
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

@dataclass()
class Dep:
    """A word in a dependency tree.

    The fields are defined by https://universaldependencies.org/format.html.
    """
    id: Text
    form: Union[Text, None]
    lemma: Union[Text, None]
    upos: Text
    xpos: Union[Text, None]
    feats: Sequence[Text]
    head: Union[Text, None]
    deprel: Union[Text, None]
    deps: Sequence[Text]
    misc: Union[Text, None]


def read_conllu(path: Text) -> Iterator[Sequence[Dep]]:
    """Reads a CoNLL-U format file into sequences of Dep objects.

    The CoNLL-U format is described in detail here:
    https://universaldependencies.org/format.html
    A few key highlights:
    * Word lines contain 10 fields separated by tab characters.
    * Blank lines mark sentence boundaries.
    * Comment lines start with hash (#).

    Each word line will be converted into a Dep object, and the words in a
    sentence will be collected into a sequence (e.g., list).

    :return: An iterator over sentences, where each sentence is a sequence of
    words, and each word is represented by a Dep object.
    """

    sentence = []
    with open(path) as infile:
        for line in infile:
            if line[0] == "#":
                continue
            line = line.strip()
            if not line:
                yield sentence
                sentence = []
                continue
            line = line.split("\t")
            for i, item in enumerate(line):
                if i == 5 or i == 8:
                    if item == "_":
                        line[i] = []
                    else:
                        line[i] = item.split("|")
                elif item == "_":
                    line[i] = None 
            word = Dep(*line)
            sentence.append(word)

class Action(Enum):
    """An action in an "arc standard" transition-based parser."""
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3


def parse(deps: Sequence[Dep],
          get_action: Callable[[Sequence[Dep], Sequence[Dep]], Action]) -> None:
    """Parse the sentence based on "arc standard" transitions.

    Following the "arc standard" approach to transition-based parsing, this
    method creates a stack and a queue, where the input Deps start out on the
    queue, are moved to the stack by SHIFT actions, and are combined in
    head-dependent relations by LEFT_ARC and RIGHT_ARC actions.

    This method does not determine which actions to take; those are provided by
    the `get_action` argument to the method. That method will be called whenever
    the parser needs a new action, and then the parser will perform whatever
    action is returned. If `get_action` returns an invalid action (e.g., a
    SHIFT when the queue is empty), an arbitrary valid action will be taken
    instead.

    This method does not return anything; it modifies the `.head` field of the
    Dep objects that were passed as input. Each Dep object's `.head` field is
    assigned the value of its head's `.id` field, or "0" if the Dep object is
    the root.

    :param deps: The sentence, a sequence of Dep objects, each representing one
    of the words in the sentence.
    :param get_action: a function or other callable that takes the parser's
    current stack and queue as input, and returns an "arc standard" action.
    :return: Nothing; the `.head` fields of the input Dep objects are modified.
    """
    queue = deque(deps)
    stack = list()

    for word in list(queue):
        if "." in word.id:
            queue.remove(word)

    while queue or len(stack) > 1:

        action = get_action(stack, queue)

        if (action.value == 1 and queue) or len(stack) < 2 :
            dep = queue.popleft()
            stack.append(dep)
        elif action.value == 2:
            stack[-2].head = stack[-1].id
            stack.pop(-2)
        else:
            stack[-1].head = stack[-2].id
            stack.pop()
      
    for dep in deps:
        if dep.head == None:
            dep.head = "0"


class Oracle:
    def __init__(self, deps: Sequence[Dep]):
        """Initializes an Oracle to be used for the given sentence.

        Minimally, it initializes a member variable `actions`, a list that
        will be updated every time `__call__` is called and a new action is
        generated.

        Note: a new Oracle object should be created for each sentence; an
        Oracle object should not be re-used for multiple sentences.

        :param deps: The sentence, a sequence of Dep objects, each representing
        one of the words in the sentence.
        """
        self.actions = []
        self.features = []

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Returns the Oracle action for the given "arc standard" parser state.

        The oracle for an "arc standard" transition-based parser inspects the
        parser state and the reference parse (represented by the `.head` fields
        of the Dep objects) and:
        * Chooses LEFT_ARC if it produces a correct head-dependent relation
          given the reference parse and the current configuration.
        * Otherwise, chooses RIGHT_ARC if it produces a correct head-dependent
          relation given the reference parse and all of the dependents of the
          word at the top of the stack have already been assigned.
        * Otherwise, chooses SHIFT.

        The chosen action should be both:
        * Added to the `actions` member variable
        * Returned as the result of this method

        Note: this method should only be called on parser state based on the Dep
        objects that were passed to __init__; it should not be used for any
        other Dep objects.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken given the reference parse
        (the `.head` fields of the Dep objects).
        """
        #Feature Extraction
        word_range = 2
        re_stack = reversed(stack)
        stack_pos = ["sPOS{}=".format(i) + dep.upos for i, dep in enumerate(re_stack) if i < word_range]
        queue_pos = ["qPOS{}=".format(i) + dep.upos for i, dep in enumerate(queue) if i < word_range]
        stack_form = ["sform{}=".format(i) + dep.form for i, dep in enumerate(re_stack) if i < word_range and dep.form]
        queue_form = ["qform{}=".format(i) + dep.form for i, dep in enumerate(queue) if i < word_range and dep.form]

        feature_list = stack_pos + queue_pos + stack_form + queue_form
        dict_f = {feature:1 for feature in feature_list}
        self.features.append(dict_f)

        if len(stack) < 2 and queue:
            self.actions.append(Action.SHIFT)
            return Action.SHIFT

        previous_head = [x.head for x in stack[:-1]]
        rest_head = [y.head for y in queue]

        if stack[-1].id in previous_head:
            self.actions.append(Action.LEFT_ARC)
            return Action.LEFT_ARC

        elif (int(stack[-1].head) < int(stack[-1].id)) and (stack[-1].id not in rest_head):
            self.actions.append(Action.RIGHT_ARC)
            return Action.RIGHT_ARC

        else:
            self.actions.append(Action.SHIFT)
            return Action.SHIFT 
    
class Classifier:
    def __init__(self, parses: Iterator[Sequence[Dep]]):
        """Trains a classifier on the given parses.

        There are no restrictions on what kind of classifier may be trained,
        but a typical approach would be to
        1. Define features based on the stack and queue of an "arc standard"
           transition-based parser (e.g., part-of-speech tags of the top words
           in the stack and queue).
        2. Apply `Oracle` and `parse` to each parse in the input to generate
           training examples of parser states and oracle actions. It may be
           helpful to modify `Oracle` to call the feature extraction function
           defined in 1, and store the features alongside the actions list that
           `Oracle` is already creating.
        3. Train a machine learning model (e.g., logistic regression) on the
           resulting features and labels (actions).

        :param parses: An iterator over sentences, where each sentence is a
        sequence of words, and each word is represented by a Dep object.
        """
        self.lg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", penalty = "l2", C = 0.1, max_iter = 300)
        self.dv = DictVectorizer()
        self.le = LabelEncoder()
        whole_feature = []
        whole_label = []
        for deps in parses:
            oracle = Oracle(deps)
            parse(deps, oracle)
            whole_feature += oracle.features
            whole_label += [action.value for action in oracle.actions]
        
        self.dv.fit(whole_feature)
        self.le.fit(whole_label)

        feature_matrix = self.dv.transform(whole_feature)
        label = self.le.transform(whole_label)

        self.lg.fit(feature_matrix, label)
        

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Predicts an action for the given "arc standard" parser state.

        There are no restrictions on how this prediction may be made, but a
        typical approach would be to convert the parser state into features,
        and then use the machine learning model (trained in `__init__`) to make
        the prediction.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken.
        """
        word_range = 2
        re_stack = reversed(stack)
        stack_pos = ["sPOS{}=".format(i) + dep.upos for i, dep in enumerate(re_stack) if i < word_range]
        queue_pos = ["qPOS{}=".format(i) + dep.upos for i, dep in enumerate(queue) if i < word_range]
        stack_form = ["sform{}=".format(i) + dep.form for i, dep in enumerate(re_stack) if i < word_range and dep.form]
        queue_form = ["qform{}=".format(i) + dep.form for i, dep in enumerate(queue) if i < word_range and dep.form]

        feature_list = stack_pos + queue_pos + stack_form + queue_form

        dict_f = {feature:1 for feature in feature_list}

        feature_matrix = self.dv.transform(dict_f)
        
        label = self.lg.predict(feature_matrix)
        action = self.le.inverse_transform(label)[0]
        
        if action == 1:
            return Action.SHIFT
        elif action == 2:
            return Action.LEFT_ARC
        else:
            return Action.RIGHT_ARC
