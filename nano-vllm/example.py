import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

text = '''电视剧《西游记》，总长达25个小时，几乎包括了百回小说《西游记》里所有精彩篇章，塑造了众多色彩绚丽的屏幕形象。 该片采用实景为主，内外景结合的拍摄方法。 既有金碧辉煌的灵宵宝殿，祥云缥缈的瑶池仙境，金镶玉砌的东海龙宫等棚内场景，又有风光倚丽的园林妙景，名山绝胜以及扬名远近的寺刹道观。 [21]
在内容方面，央视版的影视剧突出表现的是师徒四人历经九九八十一难、取得真经的经历，这与原著是一致的，但是也有小的改动。比如唐僧的“脓包”形象得到淡化，在原著中，每逢遇到危险都吓得手足无措，甚至晕死过去，在取经途中甚至萌生过“放弃”的念头，这些在影视剧的改编过程中都淡化了。最是那回眸对美丽温柔的女儿国国王的深情一望，让唐僧这个人物形象更加丰满。此外，猪八戒的形象也有所改变，影视剧中的猪八戒更可爱，原著中的猪八戒的“无能”、“惰性”都淡化了。 [22]
整部作品中，类似合成的特效有不少。那时只有一台专门的ADO特技机，剧中的特效主要是包括后期合成、抠像，以及现场特技吊威亚等等。技术不足的地方，杨洁导演就动员各工种一起想办法。比如孙悟空学习筋斗云的场景，是在杂技团的排练场拍摄完成的。先挂好蓝幕，在前面架好蹦床，借助弹跳体现“升空”。因为是有一定危险性的专业动作，所以由专业杂技演员完成，再和蓝天的镜头合到一起，呈现“腾空”的镜头。用棉花制作白云，首先是把棉花放在蓝幕前进行第一次抠像，与蓝天合成。演员在蓝幕前做出跳跃站立的动作，进行第二次与白云合成的抠像。随后在特技台上，将人物和抠好像的云彩放在一起，把握人物'''

# Test the correctness
def main():
    path = os.path.expanduser("../../models/nsa-0.5B")
    # tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1, kvcache_block_size=128)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "I like playing basketball, because",
        "I am 10 years old, how old are",
        "我最吃的食物是",
        "猫和狗",
        text
    ]

    # test prefix cache
    # prompts = [text] * 4

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()

# output. A poor 480M model was trained on 80B tokens. But it can say correct sequences?
'''
Prompt: 'I like playing basketball, because'
Completion: ' I like the game and I like the game. I like to play in the exact same situation as the game, but I also like to learn. I like the game because I like the game and I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the game. I like the game because I like the'


Prompt: 'I am 10 years old, how old are'
Completion: ' you? I’m a 60 year old college student, and I’m currently 20 years old.\nI have a very small computer with a 4.5 inch display. I have a 1540x1540 screen. The screen is in the “Home screen”, and I have a 1080x1080 screen.\nI have a 600x600 screen with a 16:9 aspect ratio.\nI am in my mid 30s, and I have been studying and studying for a year. I have been'


Prompt: '我最吃的食物是'
Completion: '巧克力，我觉得巧克力好吃，而且它的营养价值很高，我最喜欢吃的就是巧克力，我想做这道菜一定不能错过这道菜，这个巧克力不仅好吃，而且营养价值很高，让我们都垂涎欲滴，可以让我们营养变得更好。\n这道菜的制作过程还挺简单的，因为它的做法是先把巧克力切好，然后切成一小块，然后把巧克力放在容器里，放入水，盖上盖子，然后把容器放在阳光下面，在将容器盖上，然后把容器放在冰箱里，接着把容器放到锅里，把巧克力放在上面，这样，巧克力就会像巧克力一样，放在容器里，这样，巧克力就会变成'


Prompt: '猫和狗'
Completion: '的区别\n猫和狗的区别\n猫和狗\n猫和狗是两种不同的动物，它们之间存在一些区别。\n猫是一种生活在热带和亚热带地区的动物，其性格特点非常敏锐，但同时也具有攻击性和攻击性。\n而狗则是一种生活在陆地的动物，其性格通常比较温和，但同时也具有攻击性和攻击性。\n猫的体型较大，具有较小的四肢和尾巴，而狗的体型较小，具有较小的四肢和尾巴。\n猫和狗的相似之处\n猫和狗的相似之处在于它们的繁殖方式。猫是一种生活在热带和亚热带地区的动物，它们通常生活在海边或山脉附近'


Prompt: '电视剧《西游记》，总长达25个小时，几乎包括了百回小说《西游记》里所有精彩篇章，塑造了众多色彩绚丽的屏幕形象。 该片采用实景为主，内外景结合的拍摄方法。 既有金碧辉煌的灵宵宝殿，祥云缥缈的瑶池仙境，金镶玉砌的东海龙宫等棚内场景，又有风光倚丽的园林妙景，名山绝胜以及扬名远近的寺刹道观。 [21]\n在内容方面，央视版的影视剧突出表现的是师徒四人历经九九八十一难、取得真经的经历，这与原著是一致的，但是也有小的改动。比如唐僧的“脓包”形象得到淡化，在原著中，每逢遇到危险都吓得手足无措，甚至晕死过去，在取经途中甚至萌生过“放弃”的念头，这些在影视剧的改编过程中都淡化了。最是那回眸对美丽温柔的女儿国国王的深情一望，让唐僧这个人物形象更加丰满。此外，猪八戒的形象也有所改变，影视剧中的猪八戒更可爱，原著中的猪八戒的“无能”、“惰性”都淡化了。 [22]\n整部作品中，类似合成的特效有不少。那时只有一台专门的ADO特技机，剧中的特效主要是包括后期合成、抠像，以及现场特技吊威亚等等。技术不足的地方，杨洁导演就动员各工种一起想办法。比如孙悟空学习筋斗云的场景，是在杂技团的排练场拍摄完成的。先挂好蓝幕，在前面架好蹦床，借助弹跳体现“升空”。因为是有一定危险性的专业动作，所以由专业杂技演员完成，再和蓝天的镜头合到一起，呈现“腾空”的镜头。用棉花制作白云，首先是把棉花放在蓝幕前进行第一次抠像，与蓝天合成。演员在蓝幕前做出跳跃站立的动作，进行第二次与白云合成的抠像。随后在特技台上，将人物和抠好像的云彩放在一起，把握人物'
Completion: '动作的节奏，投出白云，从而表现人物的动作。 在《西游记》中，“猪八戒”这个角色有很明显的特征，尽管是唐僧的“脓包”形象，但“猪八戒”的“无能”、“惰性”在《西游记》中得到了弱化，所以这种人物形象在《西游记》中只有“猪八戒”出现。 [23]\n在人物形象方面，《西游记》中，唐僧的“脓包”形象被弱化，人物形象更加丰满，但孙悟空的形象仍然保留。孙悟空形象虽然比较复杂，但人物形象还是很丰满的。孙悟空以“猪八戒”'
'''
