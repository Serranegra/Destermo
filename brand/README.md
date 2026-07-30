# brand

Ativos de marca do Destermo.

| Arquivo | Usar quando |
|---|---|
| `destermo-icon.svg` | símbolo em verde, uso geral |
| `destermo-icon-mono.svg` | uma cor só — herda via `currentColor` |
| `destermo-lockup.svg` | símbolo + nome, horizontal |
| `favicon.svg` + `favicon.ico` | aba do navegador |
| `destermo-app-icon.svg` | fundo opaco + safe area, base dos ícones de app |
| `icon-192.png` / `icon-512.png` | manifest PWA |
| `apple-touch-icon.png` | iOS, 180×180 |
| `og-banner.png` | Open Graph e topo do README, 1200×630 |
| `destermo-icon-1024.png` | raster grande |
| `manifest.webmanifest`, `head-snippet.html` | copiar para a raiz pública |

## Paleta

```
verde    #3AA394   marca, acerto
dourado  #D3AD69   letra fora de lugar
xisto    #6E5C62   letra ausente, texto terciário
ardósia  #221C1E   fundo escuro
osso     #F5EFE9   texto sobre escuro
```

Tamanho mínimo do símbolo: **16 px**. Abaixo disso a contraforma quadrada fecha.

## Notas

- O texto de `destermo-lockup.svg` e `og-banner.svg` está como `<text>`, não como
  contorno. Fora do navegador, converta para path ou instale JetBrains Mono.
- Para regerar os PNGs e o `.ico` depois de editar um SVG: `./build.sh`
  (requer `cairosvg` e `imagemagick`).
- Estes ativos **não** estão sob a licença MIT do repositório.