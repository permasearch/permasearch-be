import pickle
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from http import HTTPStatus
from .retrieve import get_docs
from .bsbi import BSBIIndex, VBEPostings

# Create your views here.

def do_index(request):
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!
    return  JsonResponse({'status': False, 'message': "no docs found"}, status=HTTPStatus.NOT_FOUND)

@csrf_exempt
def get_document(request):
    with open("metadata/train_titles.dict", "rb") as f:
            titles_dict = pickle.load(f)
    try:
        path = request.GET["path"]
        

        with open(path,'r') as file:
            text = file.read()
        if request.method == "GET":
            pass
        else:
            return JsonResponse({'status': False})
        
        doc_id = int(path.split("/")[-1][:-4])
        data={'status': True, 'text': text, 'id': doc_id, 'path': path, 'title': titles_dict[doc_id]}
        return JsonResponse(data=data, status=HTTPStatus.OK)
    except: 
        return  JsonResponse({'status': False, 'message': "no docs found"}, status=HTTPStatus.NOT_FOUND)

@csrf_exempt
def get_serp(request):
    # try:
        query = request.GET["search"]
        data = get_docs(query)
       
        if request.method == "GET":
            pass
        else:
            return JsonResponse({'status': False})
        
        return JsonResponse({'status': True, 'data':data}, status=HTTPStatus.OK)
    # except: 
    #     return  JsonResponse({'status': False, 'message': "no docs found"}, status=HTTPStatus.NOT_FOUND)