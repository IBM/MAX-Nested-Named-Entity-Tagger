#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from core.model import ModelWrapper

from maxfw.core import MAX_API, PredictAPI, MetadataAPI
from flask_restplus import fields
from flask import request


model_wrapper = ModelWrapper()

# === Labels API

model_label = MAX_API.model('ModelLabel', {
    'id': fields.String(required=True, description='Label identifier'),
    'name': fields.String(required=True, description='Entity label'),
    'description': fields.String(required=False, description='Meaning of entity label')
})

labels_response = MAX_API.model('LabelsResponse', {
    'count': fields.Integer(required=True, description='Number of labels returned'),
    'labels': fields.List(fields.Nested(model_label), description='Entity labels that can be predicted by the model')
})

# Reference: https://www.researchgate.net/publication/10667350_GENIA_corpus-A_semantically_annotated_corpus_for_bio-textmining
tag_desc = {
    'B-*': 'Beginning of *; B- tag indicates start of a new phrase.',  # noqa
    'I-*': 'Inside; I- tag indicates inside/middle of the phrase.',  # noqa
    'L-*': 'Last; L- tag indicates last token of the phrase.',  # noqa
    'U-*': 'Unit; U- tag indicates unit-length entity',  # noqa
    '{B|I|L|U}-DNA': 'DNA',  # noqa
    'O': 'No entity type'
}


class ModelLabelsAPI(MetadataAPI):
    '''API for getting information about available entity tags'''
    @MAX_API.doc('get_labels')
    @MAX_API.marshal_with(labels_response)
    def get(self):
        '''Return the list of labels that can be predicted by the model'''
        result = {}
        result['labels'] = [{'id': l[0], 'name': l[1], 'description': tag_desc[l[1]] if l[1] in tag_desc else ''}
                            for l in model_wrapper.id_to_tag.items()]  # noqa (E741 ambiguous variable name 'l')
        result['count'] = len(model_wrapper.id_to_tag)
        return result

# === Predict API


input_example = 'The peri-kappa B site mediates human-immunodeficiency virus type 2 enhancer activation, in monocytes' \
                ' but not in T cells.'
ent_example = [
            "O",
            "B-G#DNA_domain_or_region",
            "I-G#DNA_domain_or_region",
            "L-G#DNA_domain_or_region",
            "O",
            "B-G#other_name|B-G#DNA_domain_or_region|B-G#virus",
            "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
            "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
            "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
            "I-G#other_name|I-G#DNA_domain_or_region|L-G#virus",
            "L-G#DNA_domain_or_region",
            "U-G#other_name",
            "O",
            "U-G#cell_type",
            "O",
            "O",
            "O",
            "B-G#cell_type",
            "L-G#cell_type",
            "O"
]

term_example = ["The", "peri-kappa", "B", "site", "mediates", "human-", "immunodeficiency", "virus", "type", "2",
                "enhancer", "activation", "in", "monocytes", "but", "not", "in", "T", "cells", "."]

model_input = MAX_API.model('ModelInput', {
    'text': fields.String(required=True, description='Text for which to predict entities', example=input_example)
})

model_prediction = MAX_API.model('ModelPrediction', {
    'tags': fields.List(fields.String, required=True, description='List of predicted entity tags, one per term in the input text.',  # noqa
                        example=ent_example),
    'terms': fields.List(fields.String, required=True,
                         description='Terms extracted from input text pre-processing. Each term has a corresponding predicted entity tag in the "tags" field.',  # noqa
                         example=term_example)
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'prediction': fields.Nested(model_prediction, description='Model prediction')
})


class ModelPredictAPI(PredictAPI):

    @MAX_API.doc('predict')
    @MAX_API.expect(model_input)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        '''Make a prediction given input data'''
        result = {'status': 'error'}

        j = request.get_json()
        text = j['text']
        entities, terms = model_wrapper.predict(text)
        model_pred = {
            'tags': entities,
            'terms': terms
        }
        result['prediction'] = model_pred
        result['status'] = 'ok'

        return result
