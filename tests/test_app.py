"""Module for performing unit tests"""


from app import RagChat


def test_rag_chat_gen_default() -> None:
    """ Function for testing the rag_chat_gen()"""
    test_fn = RagChat()
    test_res = test_fn.rag_chat_gen("Who won Fifa World Cup 2010?")
    test_flag = False
    if "Spain" in test_res:
        test_flag = True
    assert test_flag is True
