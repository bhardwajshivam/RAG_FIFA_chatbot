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

def test_valid_docs_default() -> None:
    """ Function for testing valid_docs()"""
    test_fn = RagChat()
    similarity_scores = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    max_similarity = 1
    assert test_fn.valid_docs(similarity_scores,max_similarity) == [5,6,7,8,9]



# End-of-file (EOF)