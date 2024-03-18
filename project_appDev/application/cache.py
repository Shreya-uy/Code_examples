from __init__ import cache
from application.models import User, List, Cards
from sqlalchemy import and_

@cache.memoize(60)
def user_data_alllists(userid):
    lists = List.query.filter(List.id ==userid).all()
    return lists

@cache.memoize(60)
def user_data_allcards(userid):
    allcards = Cards.query.join(List).filter(List.id==userid).all()
    return allcards

@cache.memoize(60)
def user_data_cards(userid,listid):
    cards=Cards.query.join(List).filter(and_(List.id==userid,List.list_id==listid)).all()
    return cards

def delete_cache_alllists(userid):
    cache.delete_memoized(user_data_alllists,userid)
    return 

def delete_cache_cards(userid,listid=None):
    cache.delete_memoized(user_data_allcards,userid)
    if listid:
        cache.delete_memoized(user_data_cards,userid,listid)
    return 